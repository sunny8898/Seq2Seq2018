#-*-coding:utf-8 -*-
import collections
import numpy as np
import tensorflow as tf
import seq2seq


class AttentionMechanism(object):
    @property
    def alignments_size(self):
        raise NotImplementedError
    @property
    def state_size(self):
        raise NotImplementedError


class _BaseAttentionMechanism(AttentionMechanism):
    def __init__(self,
                 memory, # [B, T, D]
                 probability_fn,
                 query_layer = None,
                 memory_layer = None):

        self._query_layer = query_layer
        self._memory_layer = memory_layer
        self._probability_fn = probability_fn 
        self._values = memory
        self._keys = (self.memory_layer(self._values) if self.memory_layer 
                      else self._values)
        self._alignments_size = tf.shape(self._keys)[1] # T


    @property
    def memory_layer(self):
        return self._memory_layer
    @property
    def query_layer(self):
        return self._query_layer 
    @property
    def values(self):
        return self._values
    @property
    def keys(self):
        return self._keys 
    @property
    def alignments_size(self):
        return self._alignments_size 

    def initial_alignments(self, batch_size, dtype):
        return tf.zeros([batch_size, self._alignments_size], dtype)


def _luong_score(query, keys): # query: [B, D], keys: [B, T, D]
    query = tf.expand_dims(query, 1) # [B, 1, D]
    score = tf.matmul(query, keys, transpose_b = True) # [B, [1, D] * [D, T]] -> [B, 1, T]
    score = tf.squeeze(score, [1]) # remove size 1 dimension at axis 1 -> [B, T]
    return score

class LuongAttention(_BaseAttentionMechanism):
    def __init__(self,
                 num_units,
                 memory, 
                 probability_fn = None,
                 name = "LuongAttention"):
        dtype = tf.float32
        if probability_fn is None:
            probability_fn = tf.nn.softmax 
        super(LuongAttention, self).__init__(
            query_layer = None,
            memory_layer = tf.layers.Dense(num_units, name = "memory_layer", 
                                    use_bias = False, dtype = dtype), 
            memory = memory,
            probability_fn = probability_fn)
        self._num_units = num_units
        
    def __call__(self, query):
        score = _luong_score(query, self._keys)
        alignments = self._probability_fn(score)
        return alignments


class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", 
                            "attention", # 当前的attention 
                            "time", 
                            # "alignments", # 当前的alignments，用来给monotonic那类的提供信息 
                            "alignment_history"))): # 用于之后的提取展示

    # 用于初始化时拷贝进encoder的last cell state
    def clone(self, **kwargs):
        return super(AttentionWrapperState, self)._replace(**kwargs)

def _compute_attention(attention_mechanism, cell_output, attention_layer):

    alignments = attention_mechanism(cell_output)
    expanded_alignments = tf.expand_dims(alignments, 1) # [B, 1, T]
    context = tf.matmul(expanded_alignments, attention_mechanism.values) # [B, 1, T] * [B, T, D] -> [B, 1, D]
    context = tf.squeeze(context, [1]) # [B, D]
    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1)) # [B, *]
    else:
        attention = context 

    return attention, alignments

class AttentionWrapper(tf.contrib.rnn.RNNCell):

    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size = None,
                 alignment_history = False,
                 cell_input_fn = None,
                 output_attention = True):

        if cell_input_fn is None:
            cell_input_fn = (lambda inputs, attention: 
                             tf.concat([inputs, attention], 1)) # [B, *]

        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._cell_input_fn = cell_input_fn 
        self._output_attention = output_attention
        self._alignment_history = alignment_history 


        if attention_layer_size is not None:
            self._attention_layer_size = attention_layer_size
            self._attention_size = self._attention_layer_size # attention与cell_output经过attention_layer后输出
            self._attention_layer = tf.layers.Dense(attention_layer_size, 
                                             name = "attention_layer", 
                                             use_bias = False,
                                             dtype = tf.float32)

        else:
            self._attention_layer = None
            self._attention_size = self._attention_mechanism.values.get_shape()[-1].value # 直接输出attention, 是和values中维度相同的向量

    def output_size(self):
        if self._output_attention:
            return self._attention_size 
        else:
            return self._cell.output_size

    def state_size(self):
        return AttentionWrapperState(
            cell_state = self._cell.state_size,
            attention = self._attention_size,
            time = tf.TensorShape([]),
            alignment_history = (self._attention_mechanism.alignments_size 
                                 if self.alignment_history else ()))

    def zero_state(self, batch_size, dtype):
        return AttentionWrapperState(
            cell_state = self._cell.zero_state(batch_size, dtype),
            attention = tf.zeros([batch_size, self._attention_size], dtype = dtype),
            time = tf.zeros([], dtype = tf.int32),
            alignment_history = (tf.TensorArray(dtype, size = 0, dynamic_size = True, 
                                               element_shape =
                                               tf.TensorShape(seq2seq._convert_to_shape(batch_size)).concatenate(
                                                   seq2seq._convert_to_shape(self._attention_mechanism.alignments_size))))
            if self._alignment_history else ())

    def __call__(self, inputs, state):

        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        attention, alignments = _compute_attention(self._attention_mechanism, 
                                       cell_output, 
                                       self._attention_layer)

        alignment_history = (state.alignment_history.write(state.time, alignments)
                             if self._alignment_history else ())

        next_state = AttentionWrapperState(
            cell_state = next_cell_state,
            attention = attention,
            time = state.time + 1,
            alignment_history = alignment_history)

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

