import tensorflow as tf
import numpy as np
import seq2seq 

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

class Seq2SeqModel(object):
    def __init__(self,
                 vocab_size,
                 rnn_size,
                 num_layers,
                 batch_size,
                 max_gradient_norm,
                 learning_rate,
                 forward_only):

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        
        # global_step用于保存模型时的命名
        
        self.global_step = tf.Variable(0, trainable = False)
        
        self.saver = tf.train.Saver(tf.global_variables())

        # [T, B]!
        self.encoder_inputs = tf.placeholder(shape = [None, None], dtype =
                                             tf.int32, name = "encoder_inputs")
        self.decoder_inputs = tf.placeholder(shape = [None, None], dtype =
                                             tf.int32, name = "decoder_inputs")
        self.decoder_targets = tf.placeholder(shape = [None, None], dtype =
                                              tf.int32, name =
                                              "decoder_targets")
        self.target_weights = tf.placeholder(shape = [None, None], dtype =
                                             tf.int32, name = "target_weights")


        encoder_embedding_matrix = tf.get_variable(name = "encoder_embedding_matrix",
                                                   shape = [vocab_size,
                                                            rnn_size],
                                                   dtype = tf.float32,
                                                   initializer =
                                                   tf.truncated_normal_initializer(
                                                       mean = 1.0, stddev = 0.0), 
                                                   )
        decoder_embedding_matrix = tf.get_variable(name = "decoder_embedding_matrix",
                                                   shape = [vocab_size,
                                                            rnn_size],
                                                   dtype = tf.float32,
                                                   initializer =
                                                   tf.truncated_normal_initializer(
                                                       mean = 0.0, stddev = 0.1), 
                                                   )
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding_matrix, 
                                                         self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding_matrix, 
                                                         self.decoder_inputs)

        with tf.variable_scope("encoder"):
            encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size) 
                                                        for _ in range(num_layers)])

            self.encoder_all_outputs, self.encoder_final_state = seq2seq.dynamic_rnn(
                encoder_cell, self.encoder_inputs_embedded, dtype = tf.float32)



    def step(self, sess):
        feed_dict = {self.encoder_inputs.name : [[0, 2], [2, 3], [2, 2], [1, 0]]}
        (outputs, final_state) = sess.run([self.encoder_all_outputs, self.encoder_final_state], feed_dict = feed_dict)
        return (outputs, final_state)

            

model = Seq2SeqModel(4, 5, 1, 2, 1, 0.9, True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    (outputs, final_state) = model.step(sess)
    print("from my dyanmic_rnn")
    print(outputs)
    print(final_state)








