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

    """
    state_size = cell.state_size

    inputs_shape = array_ops.shape(inputs)
    time_steps = inputs_shape[0]
    batch_size = inputs_shape[1]

    output_size = cell.output_size

    element_shape = tensor_shape.TensorShape(_convert_to_shape(batch_size)).concatenate(_convert_to_shape(output_size))
    print(element_shape)


    time = tf.constant(0)

    if initial_state is not None:
        state = initial_state 
    else:
        state = cell.zero_state(batch_size, dtype)


    print("so far so good")
    
    output_ta = tf.TensorArray(dtype = tf.float32, size = time_steps,
                               element_shape = element_shape)

    def _time_step(time, output_ta_t, state):
        input_t = inputs[time]
        output, next_state = cell(input_t, state)
        output_ta_t = output_ta_t.write(time, output)
        return (time + 1, output_ta_t, next_state)

    def _cond(time, output_ta_t, state):
        loop_bound = time_steps
        return time < loop_bound


    _, outputs, final_state = control_flow_ops.while_loop(
        cond = _cond,
        body = _time_step,
        loop_vars = (time, output_ta, state))
    
    outputs = outputs.stack()

    return (outputs, final_state)


    

