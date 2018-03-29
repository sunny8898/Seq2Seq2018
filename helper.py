import abc
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops


def unstack(inputs):
    return tf.TensorArray(
        dtype = inputs.dtype, 
        size = array_ops.shape(inputs)[0], 
        element_shape = array_ops.shape(inputs)[1 : ]).unstack(inputs)

def _convert_to_shape(shape): 
    if isinstance(shape, ops.Tensor):
        return tensor_shape.as_shape(tensor_util.constant_value(shape))
    else:
        return shape

@six.add_metaclass(abc.ABCMeta)
class Helper(object):
    @abc.abstractproperty 
    def batch_size(self):
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
        self.zero_inputs = array_ops.zeros_like(inputs[0, : ])
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

    def initialize(self):
        finished = math_ops.equal(0, self._seq_lengths)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(all_finished, 
                                            lambda: self._zero_inputs,
                                            lambda: self._inputs_ta.read(0))
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        # outputs : [B, D]
        return math_ops.cast(math_ops.argmax(outputs, axis = -1), dtypes.int32)
        # [B]

    def next_inputs(self, time, outputs, state, sample_ids):
        next_time = time + 1
        finished = math_ops.greater(next_time, self._seq_lengths - 1)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(all_finished,
                                            lambda: self._zero_inputs,
                                            lambda: self._inputs_ta.read(next_tinext_time))
        return (finished, next_inputs, state)

    

class GreedyEmbeddingHelper(Helper):
    def __init__(self, embedding_matrix, start_tokens, end_token):
        """Initializer.
        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`. The returned tensor
            will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
            scalar.
        """
        self._embedding_fn = (lambda ids: tf.nn.embedding_lookup(embedding_matrix, ids))
        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype = dtypes.int32, name = "start_tokens")
        self._batch_size = array_ops.size(start_tokens)
        self._end_token = ops.convert_to_tensor(
            end_token, dtype = dtypes.int32, name = "end_token")
        self._start_inputs = self._embedding_fn(self._start_tokens)

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state):
        return math_ops.cast(math_ops.argmax(outputs, axis = -1), dtypes.int32)

    def next_inputs(self, time, outputs, state, sample_ids):
        next_time = time + 1
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond( all_finished, 
                                            lambda: self._start_inputs, 
                                            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):

    """An RNN Decoder abstract interface object.
  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
  """ 
  @property
  def batch_size(self):
      raise NotImplementedError
  @property
  def output_size(self):
      raise NotImplementedError
  @property
  def output_type(self):
      raise NotImplementedError

  @abc.abstractmethod
  def initialize(self):
      """Returns (finished, initial_inputs, initial_state)"""
      raise NotImplementedError
  @abc.abstractmethod
  def step(self, time, inputs, state):
      """Returns (outputs, next_state, next_inputs, finished)"""
      raise NotImplementedError


class BasicDecoder(Decoder):
    def __init__(self, cell, helper, initial_state, output_layer = None):
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer 
    
    @property
    def batch_size(self):
        return self._helper.batch_size 

    @property
    def output_size(self):
        output_size = self._output_size
        if self._output_layer is None:
            return output_size
        else:
            layer_output_shape = self._output_layer.compute_output_shape(output_size)
      return layer_output_shape[1 : ]

    def initialize(self):
        return self._helper.initialize() + (self._initial_state, )

    def step(self, time, inputs, state):
        outputs, state = self._cell(inputs, state)
        if self._output_layer is not None:
            outputs = self._output_layer(outputs)
        sample_ids = self._helper.sample(time, outputs, state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            self, time, outputs, state, sample_ids)
        return (outputs, next_state, next_inputs, finished)


def dynamic_decode(decoder):
    (finished, initial_inputs, initial_state) = decoder.initialize()
    initial_time = tf.constant(0, dtype = dtypes.int32)

    output_ta = tf.TensorArray(
        dtype = tf.float32, size = 0, dynamic_size = True,
        element_shape = decoder.output_size)

    def _cond(time, finished, inputs, state, output_ta):
        all_finished = math_ops.reduce_all(finished)
        return math_ops.logical_not(all_finished)
    
    def _body(time, finished, inputs, state, output_ta):
        (outputs, next_state, next_inputs, finished) = decoder.step(time, inputs, state)
        output_ta = output_ta.write(time, outputs)
        return (time + 1, finished, next_inputs, next_state, output_ta)

    _, _, _, final_state, output_ta = control_flow_ops.while_loop(
        cond = _cond, 
        body = _body,
        loop_vars = (time, finished, inputs, state, output_ta))

    final_outputs = output_ta.stack()

    return (final_outputs, final_state)
    















