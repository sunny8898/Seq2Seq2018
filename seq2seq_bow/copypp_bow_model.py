#-*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import seq2seq 
import data_utils
import attention
import seq2seq

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util




setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

class Seq2SeqModel(object):
    def __init__(self,
                 vocab_size,
                 rnn_size,
                 encoder_layers,
                 decoder_layers,
                 attention_depth,
                 max_gradient_norm,
                 learning_rate,
                 copy_dict):

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.attention_depth = attention_depth
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        
        # global_step用于保存模型时的命名
        
        self.global_step = tf.Variable(0, trainable = False)

        self.print_ops = []

        with tf.variable_scope("cnt", reuse = tf.AUTO_REUSE):
            self.global_cnt = tf.get_variable(name = "global_cnt",
                                              shape = [],
                                              dtype = tf.int32,
                                              initializer =
                                              tf.constant_initializer(0),
                                              trainable = False)

        # copypp list
        self.copy_dict = copy_dict

        copy_indices_lst = []
        copy_values_lst = []
        for k, vs in copy_dict.items():
            for v in vs:
                copy_indices_lst.append([v, k])
                copy_values_lst.append(1)
        copy_indices = tf.constant(copy_indices_lst, name = "copy_indices")
        copy_values = tf.constant(copy_values_lst, dtype = tf.float32, name = "copy_values")
        copy_shape = tf.constant([vocab_size, vocab_size], dtype = tf.int64, name = "copy_shape")

        print(copy_indices.shape)

        copy_mat = tf.SparseTensor(indices = copy_indices_lst, values = copy_values, dense_shape = copy_shape)

        tmp = tf.sparse_tensor_to_dense(copy_mat)

        self.print_ops.append(tf.Print(tmp, [tmp], "copy_mat:",
                                       summarize = 1000))





        


        self.init = tf.global_variables_initializer()

        # [T, B]!
        self.encoder_inputs = tf.placeholder(shape = [None, None], dtype =
                                             tf.int32, name = "encoder_inputs")
        self.decoder_inputs = tf.placeholder(shape = [None, None], dtype =
                                             tf.int32, name = "decoder_inputs")
        self.decoder_targets = tf.placeholder(shape = [None, None], dtype =
                                              tf.int32, name =
                                              "decoder_targets")
        self.target_weights = tf.placeholder(shape = [None, None], dtype =
                                             tf.float32, name = "target_weights")

        self.encoder_embedding_matrix = tf.get_variable(name = "encoder_embedding_matrix",
                                                   shape = [vocab_size,
                                                            rnn_size],
                                                   dtype = tf.float32,
                                                   initializer =
                                                   tf.truncated_normal_initializer(
                                                       mean = 0.0, stddev = 0.1), 
                                                   )
        self.decoder_embedding_matrix = tf.get_variable(name = "decoder_embedding_matrix",
                                                   shape = [vocab_size,
                                                            rnn_size],
                                                   dtype = tf.float32,
                                                   initializer =
                                                   tf.truncated_normal_initializer(
                                                       mean = 0.0, stddev = 0.1), 
                                                   )
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.encoder_embedding_matrix, 
                                                              self.encoder_inputs, 
                                                              name = "encoder_inputs_embedded")
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.decoder_embedding_matrix, 
                                                              self.decoder_inputs, 
                                                              name = "decoder_inputs_embedded")

        with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
            """
            encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size) 
                                                        for _ in range(num_layers)])

            self.encoder_all_outputs, self.encoder_final_state = seq2seq.dynamic_rnn(
                encoder_cell, self.encoder_inputs_embedded, dtype = tf.float32)
            """
            self.encoder_all_outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                [tf.contrib.rnn.BasicLSTMCell(rnn_size) for _ in range(encoder_layers)], 
                [tf.contrib.rnn.BasicLSTMCell(rnn_size) for _ in range(encoder_layers)], 
                self.encoder_inputs_embedded, dtype = tf.float32, time_major = True) 
            # dtype: the data type of the initial state, should be provided if either of the initial states are not provided

            self.print_ops.append(tf.Print(tf.shape(states_fw),
                                           [tf.shape(states_fw)], "states_fw = "))
            self.print_ops.append(tf.Print(tf.shape(states_fw[0]),
                                           [tf.shape(states_fw[0])],
                                           "states_fw[0] = "))


            trans_state_layer = tf.layers.Dense(rnn_size)

            final_state = tf.concat([states_fw[-1], states_bw[-1]], -1) # [B, D] + [B, D] -> [B, 2 * D]
            # Actually states_fw[-1] is LSTMStateTuple(c = [B, D], h = [B, D]),
            # after tf.concat the names(c, h) are lost -> [2, B, D]
            final_state = trans_state_layer(final_state) # [2, B, 2 * D] -> [2, B, D]
            self.encoder_final_state = tf.contrib.rnn.LSTMStateTuple(final_state[0], 
                                                        final_state[1]) 
            # [2, B, D] -> LSTMStateTuple(c = [B, D], h = [B, D]




        with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
            decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)
                                                        for _ in range(decoder_layers)])
            memory = tf.transpose(self.encoder_all_outputs, perm = [1, 0, 2]) # [T, B, D] -> [B, T, D]
            batch_size = tf.shape(self.encoder_all_outputs)[1] # B

            enc_out_size = tf.shape(memory)
            self.print_ops.append(tf.Print(enc_out_size, [enc_out_size],
                                           "enc_out_size = "))

    
            attention_mechanism = attention.LuongAttention(
                num_units = rnn_size,
                memory = memory)
            
            attn_cell = attention.AttentionWrapper(decoder_cell, 
                                                   attention_mechanism, 
                                                   attention_layer_size = attention_depth)

            # cell_outputs从[B, D] -> [B, vocab_size]
            fc_layer = tf.layers.Dense(vocab_size) 

            training_helper = seq2seq.TrainingHelper(self.decoder_inputs_embedded)

            """
            d_size = tf.shape(self.encoder_final_state)
            self.print_ops.append(tf.Print(d_size, [d_size], "d_size = "))
            """

            initial_cell_state = decoder_cell.zero_state(batch_size, tf.float32)
            tmp_list = [initial_cell_state[i] 
                        for i in range(len(initial_cell_state))]
            tmp_list[0] = self.encoder_final_state
            initial_cell_state = tuple(tmp_list)


            """
            e_size = tf.shape(initial_cell_state)
            self.print_ops.append(tf.Print(e_size, [e_size], "e_size = "))
            """


            training_decoder = seq2seq.BasicDecoder(
                cell = attn_cell, 
                helper = training_helper, 
                initial_state = attn_cell.zero_state(batch_size, 
                                                     tf.float32).clone(cell_state = initial_cell_state),
                output_layer = fc_layer) 
                                                    

            self.decoder_outputs, self.decoder_final_state = seq2seq.dynamic_decode(training_decoder)

            self.decoder_logits = self.decoder_outputs.rnn_output # [T, B, V]


        with tf.variable_scope("training", reuse = tf.AUTO_REUSE):

            self.targets_one_hot = tf.one_hot(self.decoder_targets, depth = vocab_size, 
                                    axis = -1, dtype = tf.float32) # [T, B, V]

            def _convert_to_shape(shape): 
                if isinstance(shape, ops.Tensor):
                    return tensor_shape.as_shape(tensor_util.constant_value(shape))
                else:
                    return shape

            def expand(one_hot):

                one_hot = tf.reshape(one_hot, [-1, vocab_size])

                one_hot_t = tf.transpose(one_hot, [1, 0])

                return tf.add(one_hot, tf.transpose(
                        tf.sparse_tensor_dense_matmul(copy_mat,
                                                      one_hot_t), [1, 0]))


            self.targets_expanded = tf.reshape(expand(self.targets_one_hot), [-1, batch_size, vocab_size])

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels = self.targets_one_hot,
                logits = self.decoder_logits)

            self.seq_loss = tf.reduce_sum(stepwise_cross_entropy * self.target_weights) / tf.reduce_sum(self.target_weights)



            tmp = tf.reduce_sum(self.decoder_logits, 0) # [T, B, V] -> [B, V]

            self.print_ops.append(tf.Print(tmp, [tmp], "reduced:\n", summarize = 10000))

            """
            self.print_ops.append(tf.Print(self.targets_one_hot, 
                                           [tf.shape(self.targets_one_hot), self.targets_one_hot], 
                                           "one-hot:\n", summarize = 10000)) 
            self.print_ops.append(tf.Print(self.targets_expanded, 
                                           [self.targets_expanded], 
                                           "expanded:\n", summarize = 10000)) 
            """



            tmp = tf.reduce_sum(self.targets_expanded * tmp, -1) 

            self.print_ops.append(tf.Print(tmp, [tf.shape(tmp), tmp], 
                                           "after multi with expanded:", summarize = 10000)) 
            tmp = tf.sigmoid(tmp)

            self.print_ops.append(tf.Print(tmp, [tf.shape(tmp), tmp], "after sigmoid:\n", summarize = 10000)) 

            self.bow_loss = tf.reduce_sum(-tf.log(tmp + 0.000001) * self.target_weights) / tf.reduce_sum(self.target_weights)
            
            self.lbd = tf.placeholder(shape = [], dtype = tf.float32, name = "loss_lambda")

            with tf.control_dependencies([self.global_cnt.assign(
                self.global_cnt + batch_size)]):
                # self.loss = self.seq_loss + self.bow_loss * self.lbd
                self.loss = self.bow_loss * self.lbd

            params = tf.trainable_variables()

            #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            optimizer = tf.train.AdamOptimizer()

            gradients = tf.gradients(self.loss, params)

            clipped_grad, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            self.training_op = (optimizer.apply_gradients(zip(clipped_grad, params),
                                                      self.global_step))
            # self.global_step -> will be incremented after each update

        with tf.variable_scope("inference", reuse = tf.AUTO_REUSE):

            num_seq_to_decode = array_ops.shape(self.encoder_inputs)[1] # [T, B]

            start_tokens = tf.tile([data_utils.GO_ID], [num_seq_to_decode])

            end_token = data_utils.EOS_ID 

            decoding_helper = seq2seq.GreedyEmbeddingHelper(
                self.decoder_embedding_matrix, start_tokens, end_token)

            greedy_decoder = seq2seq.BasicDecoder(
                cell = attn_cell, 
                helper = decoding_helper, 
                initial_state = attn_cell.zero_state(batch_size,
                                                     tf.float32).clone(cell_state = initial_cell_state),
                output_layer = fc_layer) 
            
            self.infer_results, self.infer_final_state = seq2seq.dynamic_decode(greedy_decoder)
            self.infer_results = self.infer_results.sample_id

        self.saver = tf.train.Saver(tf.global_variables())



    def test_step(self, sess):
        feed_dict = {self.encoder_inputs.name : [[1, 2], [2, 3], [2, 2], [1, 1]],
                     self.decoder_inputs.name : [[0, 0], [2, 3], [3, 1], [3, 3], [2, 2]], 
                     self.decoder_targets.name : [[2, 3], [3, 1], [3, 3], [2, 2], [0, 0]],
                     self.target_weights.name : [[1, 1], [1, 1], [1, 1], [1,
                                                                          1],
                                                 [1, 1]],
                     self.lbd.name: 10}
        loss, _, __ = sess.run([self.loss, self.training_op] + [self.print_ops], feed_dict = feed_dict)
        return loss

    def test_decode(self, sess):
        feed_dict = {self.encoder_inputs.name : [[1, 1, 2], [2, 2, 3], [2, 2, 2], [1, 1, 1]]}
        results = sess.run([self.infer_results], feed_dict = feed_dict)
        return results

    def step(self, sess, 
             encoder_inputs, decoder_inputs, decoder_targets, target_weights,
             lbd):
        feed_dict = {self.encoder_inputs.name: encoder_inputs,
                     self.decoder_inputs.name: decoder_inputs,
                     self.decoder_targets.name: decoder_targets,
                     self.target_weights.name: target_weights,
                     self.lbd.name: lbd}
        loss, _ = sess.run([self.loss, self.training_op], feed_dict = feed_dict)
        return loss

    def predict(self, sess, encoder_inputs):
        feed_dict = {self.encoder_inputs.name: encoder_inputs}
        results = sess.run([self.infer_results], feed_dict = feed_dict)
        return results

    def get_cnt(self, sess):
        return sess.run([self.global_cnt])[0]





def self_test():
    copy_dict = {4: [5]}
    model = Seq2SeqModel(10, 8, 3, 2, 8, 0.1, 0.9, copy_dict)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000):
            loss = model.test_step(sess)
            if _ % 100 == 0:
               print(loss)
        results = model.test_decode(sess)
        print(results)


# self_test()







