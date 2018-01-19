# -*- coding:utf-8 -*-  

from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import variable_scope, math_ops, array_ops
from tensorflow.python.ops import rnn, rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

'''
attention烂坑了...

def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function

def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):

  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

    state = initial_state

    def attention(query):
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat_v2(query_list, 1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [
        array_ops.zeros(
            batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
    ]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + attns, input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state


def embedding_attention(encoder_inputs, decoder_inputs, cell,
                        num_encoder_symbols, num_decoder_symbols,
                        embedding_size, output_projection, feed_previous):
    with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
        dtype = scope.dtype
        # encoder
        encoder_cell = core_rnn_cell.EmbeddingWrapper(cell, embedding_classes = num_encoder_symbols,
                                                      embedding_size = embedding_size)
        encoder_outputs, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype = dtype)

        # decoder
        output_size = cell.output_size
        proj_biases = ops.convert_to_tensor(output_projection[1])
        embedding = variable_scope.get_variable("embedding", [num_decoder_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        embedding_inputs = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(embedding_inputs, encoder_state, attention_states,
                                 cell, output_size = output_size, loop_function = loop_function)

'''

def seq_loss(logits, targets, target_weights, softmax_loss_function):
    '''
    计算交叉熵损失函数
    Args:
        logits, targets, target_weights: 目标输出与给target的权重
        softmax_loss_function: 做softmax的损失函数
    Returns:
        损失函数的标量值
    '''
    with ops.name_scope(None, "sequence_loss", logits + targets + target_weights):
        cur = []
        for(_l, _t, _w) in zip(logits, targets, target_weights):
            cur.append(_w * softmax_loss_function(logits = _l, labels = _t))

        loss = math_ops.reduce_sum(math_ops.add_n(cur) / math_ops.add_n(target_weights))

    return loss / math_ops.cast(array_ops.shape(targets[0])[0], loss.dtype)


def model_with_buckets(encoder_inputs, decoder_inputs, targets, target_weights, 
                       buckets, seq2seq, softmax_loss_function):
    '''
    带bucket的s2s
    Args:
        encoder_inputs, decoder_inputs: encoder端和decoder端的输入
        targets, target_weights: 目标输出与给target的权重
        buckets: 由外层model传进来的输入输出的组列表
        seq2seq: seq2seq函数seq2seq_f，传入参数为en和de端的输入，产生de端的输出和
                 对应的状态
        softmax_loss_function: 做softmax的损失函数
        
    Returns:
        output: buckets大小的列表
        loss: 损失函数
    '''
    output, loss = [], []
    with ops.name_scope(None, "model_with_buckets", 
                        encoder_inputs + decoder_inputs + targets + target_weights):
        for i, bucket in enumerate(buckets):
            reuse = True
            if i == 0:
                reuse = None
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse = reuse):
                cur, _ = seq2seq(encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1]])
                output.append(cur)
                loss.append(seq_loss(output[-1], targets[:bucket[1]],target_weights[:bucket[1]],
                                     softmax_loss_function)
                           )
    return output, loss