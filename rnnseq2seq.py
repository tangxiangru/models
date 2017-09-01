def model_inputs():
    input = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    # training parameters
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    # sequence length parameters
    target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
    max_target_sequence_length = tf.reduce_max(target_sequence_length)
    source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")

    return (input, targets, learning_rate, keep_prob, target_sequence_length, \
            max_target_sequence_length, source_sequence_length)


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    x = tf.strided_slice(target_data, [0,0], [batch_size, -1], [1,1])
    y = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), x], 1)
    return y


# Encoding
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, source_sequence_length, source_vocab_size, encoding_embedding_size):
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer=output_layer)
    dec_outputs, dec_state = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_summary_length)
    return dec_outputs

## Decoding Inference
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob):
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
        start_tokens, end_of_sequence_id)
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer=output_layer)
    dec_outputs, dec_state = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True,
                                        maximum_iterations=max_target_sequence_length)
    return dec_outputs

## Decoding Layer
from tensorflow.python.layers import core as layers_core
def decoding_layer(dec_input, encoder_state, target_sequence_length, max_target_sequence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, decoding_embedding_size):
    # embedding target sequence
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    # construct decoder lstm cell
    dec_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.LSTMCell(rnn_size) \
        for _ in range(num_layers) ])
    output_layer = layers_core.Dense(target_vocab_size,
                                    kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    with tf.variable_scope("decoding") as decoding_scope:
        dec_outputs_train = decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                             target_sequence_length, max_target_sequence_length,
                             output_layer, keep_prob)
    # decoder inference
    start_of_sequence_id = target_vocab_to_int["<GO>"]
    end_of_sequence_id = target_vocab_to_int["<EOS>"]
    with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        dec_outputs_infer = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                             end_of_sequence_id, max_target_sequence_length,
                             target_vocab_size, output_layer, batch_size, keep_prob)
    # rerturn
    return dec_outputs_train, dec_outputs_infer

# Seq2seq
def seq2seq_model(input_data, target_data, keep_prob, batch_size, source_sequence_length, target_sequence_length, max_target_sentence_length, source_vocab_size, target_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    enc_output, enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,
                   enc_embedding_size)
    # process target data
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    # embedding and decoding
    dec_outputs_train, dec_outputs_infer = decoding_layer(dec_input, enc_state,
                   target_sequence_length, tf.reduce_max(target_sequence_length),
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, dec_embedding_size)
    return dec_outputs_train, dec_outputs_infer

# Build Graph
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# Training
""" DON'T MODIFY ANYTHING IN THIS CELL """
def get_accuracy(target, logits):
    """ Calculate accuracy """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):

        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})


                train_acc = get_accuracy(target_batch, batch_train_logits)


                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)


                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
