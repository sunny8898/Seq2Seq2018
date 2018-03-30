import abc
import six
import collections
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import ops

from tensorflow.python.util import nest

def _convert_to_shape(shape): 
    if isinstance(shape, ops.Tensor):
        return tensor_shape.as_shape(tensor_util.constant_value(shape))
    else:
        return shape

def dynamic_rnn(cell,
                inputs,
                initial_state = None,
                dtype = None):
    """
    Args:
        cell: RNNCell
        inputs: [T, B, D]
        initial_state: RNNCell的初始状态
    Returns:
        outputs: 所有输出
        final_state: RNNCell的最终状态
    """

    inputs_shape = array_ops.shape(inputs) # type: ops.Tensor
    time_steps = inputs_shape[0]
    batch_size = inputs_shape[1]

    output_size = cell.output_size

    output_element_shape = tensor_shape.TensorShape(_convert_to_shape(batch_size)).concatenate(output_size) # [B * output_size] 
    

    time = tf.constant(0)

    if initial_state is not None:
        state = initial_state 
    else:
        state = cell.zero_state(batch_size, dtype = dtype)

    input_ta = tf.TensorArray(dtype = inputs.dtype, size = time_steps,
                              element_shape = inputs.shape[1 : ]) # [T, B, D]
    input_ta = input_ta.unstack(inputs)
    output_ta = tf.TensorArray(dtype = tf.float32, size = time_steps,
                               element_shape = output_element_shape) # [T, B, D]

    def _time_step(time, output_ta_t, state): # while_loop的循环体
        input_t = input_ta.read(time) # 读当前步的输入
        output, next_state = cell(input_t, state) # 产生输出与新的状态
        output_ta_t = output_ta_t.write(time, output) # 写入输出
        return (time + 1, output_ta_t, next_state)

    def _cond(time, output_ta_t, state): # while_loop的循环中止判定函数
        loop_bound = time_steps
        return time < loop_bound

    _, outputs, final_state = control_flow_ops.while_loop(
        cond = _cond,
        body = _time_step,
        loop_vars = (time, output_ta, state))
    
    final_outputs = outputs.stack() # 将TensorArray变成一个Tensor

    return (final_outputs, final_state)


    
def unstack(inputs):
    return tf.TensorArray(
        dtype = inputs.dtype, 
        size = array_ops.shape(inputs)[0], 
        element_shape = inputs.shape[1 : ]).unstack(inputs)

@six.add_metaclass(abc.ABCMeta)
class Helper(object):
    @abc.abstractproperty 
    def batch_size(self):
        raise NotImplementedError
    @abc.abstractproperty
    def sample_ids_shape(self):
        raise NotImplementedError
    @abc.abstractmethod
    def initialize(self):
        """Returns (initial_finished, initial_inputs) """
        raise NotImplementedError
    @abc.abstractmethod
    def sample(self, time, outputs, state):
        """Returns sample_ids"""
        raise NotImplementedError
    @abc.abstractmethod
    def next_inputs(self, time, outputs, state, sample_ids):
        """Returns (finished, next_inputs, next_state)"""
        raise NotImplementedError

class TrainingHelper(Helper):
    def __init__(self, inputs):
        # [T, B, D]
        inputs = ops.convert_to_tensor(inputs, name="inputs")
        inputs_shape = array_ops.shape(inputs)
        self._batch_size = inputs_shape[1]
        self._time_step = inputs_shape[0]
        # 这里直接用padding把batch里的所有序列都弄到一样长了，不过为了接口统一还是要做一个[B]大小的tensor来放序列长度
        self._seq_lengths = array_ops.tile([self._time_step], [self._batch_size])
        self._inputs = inputs
        self._inputs_ta = unstack(inputs)
        self._zero_inputs = array_ops.zeros_like(inputs[0, : ])
    @property
    def inputs(self):
        return self._inputs
    @property
    def time_step(self):
        return self._time_step
    @property
    def seq_lengths(self):
        return self._seq_lengths
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    def initialize(self):
        finished = math_ops.equal(0, self._seq_lengths)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(all_finished, 
                                            lambda: self._zero_inputs,
                                            lambda: self._inputs_ta.read(0))
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        # outputs : [B, D]
        return math_ops.cast(math_ops.argmax(outputs, axis = -1), tf.int32)
        # [B]

    def next_inputs(self, time, outputs, state, sample_ids):
        next_time = time + 1
        finished = math_ops.greater(next_time, self._seq_lengths - 1)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(all_finished,
                                            lambda: self._zero_inputs,
                                            lambda: self._inputs_ta.read(next_time))
        return (finished, next_inputs, state)

    

class GreedyEmbeddingHelper(Helper):
    def __init__(self, embedding_matrix, start_tokens, end_token):
        self._embedding_fn = (lambda ids: tf.nn.embedding_lookup(embedding_matrix, ids))
        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype = tf.int32, name = "start_tokens")
        self._batch_size = array_ops.size(start_tokens)
        self._end_token = ops.convert_to_tensor(
            end_token, dtype = tf.int32, name = "end_token")
        self._start_inputs = self._embedding_fn(self._start_tokens)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    def initialize(self):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state):
        return math_ops.cast(math_ops.argmax(outputs, axis = -1), tf.int32)

    def next_inputs(self, time, outputs, state, sample_ids):
        next_time = time + 1
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(all_finished, 
                                            lambda: self._start_inputs, 
                                            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):

  @abc.abstractproperty
  def batch_size(self):
      raise NotImplementedError
  @abc.abstractproperty
  def output_size(self):
      raise NotImplementedError
  @abc.abstractproperty
  def output_dtype(self):
      raise NotImplementedError

  @abc.abstractmethod
  def initialize(self):
      """Returns (finished, initial_inputs, initial_state)"""
      raise NotImplementedError
  @abc.abstractmethod
  def step(self, time, inputs, state):
      """Returns (outputs, next_state, next_inputs, finished)"""
      raise NotImplementedError

# 在decode的时候，我们不仅希望得到logits(训练时)，也希望得到预测时作为输入的真·预测输出，用一个namedtuple组织起来一同作为输出
class BasicDecoderOutput(collections.namedtuple("BasicDecoderOutput",
                                                ("rnn_output", "sample_id"))):
    pass

class BasicDecoder(Decoder):
    def __init__(self, cell, helper, initial_state, output_layer = None):
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer 
    
    @property
    def batch_size(self):
        return self._helper.batch_size 

    def _rnn_output_size(self):
        # 单个output的size, batch_size在之后加
        if self._output_layer is None:
            return self._cell.output_size
        else:
            return self._output_layer.units

    @property
    def output_size(self):
        return BasicDecoderOutput(rnn_output = self._rnn_output_size(),
                                  sample_id = self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        return BasicDecoderOutput(
            nest.map_structure(lambda _: tf.float32, self._rnn_output_size()),
            tf.int32)


    def initialize(self):
        return self._helper.initialize() + (self._initial_state, )

    def step(self, time, inputs, state):
        outputs, state = self._cell(inputs, state)
        if self._output_layer is not None:
            outputs = self._output_layer(outputs)
        sample_ids = self._helper.sample(time, outputs, state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(time, outputs, state, sample_ids)
        # 保留logits以及作为输入的信息sample_ids
        outputs = BasicDecoderOutput(outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)


def dynamic_decode(decoder):
    (finished, initial_inputs, initial_state) = decoder.initialize()
    initial_time = tf.constant(0, dtype = tf.int32)

    def _create_ta(size, dtype):
        return tf.TensorArray(
            dtype = dtype,
            size = 0,
            dynamic_size = True,
            element_shape = tensor_shape.TensorShape(
                _convert_to_shape(decoder.batch_size).concatenate(size)))

    output_ta = nest.map_structure(
        _create_ta, 
        decoder.output_size, 
        decoder.output_dtype)

    def _cond(time, finished, inputs, state, output_ta):
        all_finished = math_ops.reduce_all(finished)
        return math_ops.logical_not(all_finished)
    
    def _body(time, finished, inputs, state, output_ta):
        (outputs, next_state, next_inputs, finished) = decoder.step(time, inputs, state)
        output_ta = nest.map_structure(lambda ta, out: ta.write(time, out), 
                                       output_ta, outputs)
        return (time + 1, finished, next_inputs, next_state, output_ta)

    _, _, _, final_state, output_ta = control_flow_ops.while_loop(
        cond = _cond, 
        body = _body,
        loop_vars = (initial_time, finished, initial_inputs, initial_state, output_ta))

    final_outputs = nest.map_structure(lambda ta: ta.stack(), output_ta)

    return (final_outputs, final_state)
 
