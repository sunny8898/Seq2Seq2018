import tensorflow as tf
import numpy as np

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

def seq2seq_with_buckets(encoder_inputs, decoder_inputs, 
                         targets, target_weights, buckets, 
                         seq2seq_f, softmax_loss_function):
    return tf.contrib.legacy_seq2seq.model_with_buckets(encoder_inputs, decoder_inputs, 
                                                        targets, target_weights, buckets, 
                                                        seq2seq_f, softmax_loss_function = softmax_loss_function)

class Seq2SeqModel(object):
    
    def __init__(self,
                 vocab_size,
                 rnn_size,
                 num_layers,
                 buckets,
                 batch_size,
                 max_gradient_norm,
                 num_sampled, 
                 learning_rate,
                 forward_only):
        '''
        Args:
            与网络有关的超参数：
            vocab_size: 输入的维数
            rnn_size：隐层维数，它是embedding的维数，也是cell中状态的维数
            num_layers: rnn的层数
            与训练/图有关的超参数：
            buckets: 若干组不同的输入长度，将会建成若干组对应的图
            batch_size: 每个batch的大小
            max_gradient: 执行gradient clipping时的参数
            learning_rate: 学习率
            num_sampled: 计算loss时，不对整个vocab_size大小的output进行计算，而是采target中出现过的labels以及随机的num_samples个位置，计算loss
            forward_only: 是否只有前向没有后向
        '''
        
        # 赋值
        
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.buckets = buckets
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.num_sampled = num_sampled
        
        self.learning_rate = learning_rate
        
        # global_step用于保存模型时的命名
        
        self.global_step = tf.Variable(0, trainable = False)
        
        # 在计算loss/预测时，需要把rnn_size的输出投影到vocab_size上，增加一个投影层
        
        proj_w = tf.get_variable("proj_w", [rnn_size, vocab_size])
        proj_b = tf.get_variable("proj_b", [vocab_size])
        
        w_t = tf.transpose(proj_w)
        
        output_proj = (proj_w, proj_b)
        
        # 用于计算之后的sampled_softmax_loss
        
        def sampled_loss(labels, logits):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(w_t, proj_b, labels = labels, inputs = logits, num_sampled = num_sampled, num_classes = vocab_size)
        
        # 构建网络里的单个cell
        
        cell =  tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_layers)])
        
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, 
                decoder_inputs,
                cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=rnn_size,
                output_projection=output_proj,
                feed_previous=do_decode)
        
        # 为encoder_inputs, decoder_inputs, target_weights建立placeholders
        
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.targets = []
        
        # 这里的shape其实是后续的batch_size，如果不这样设置，去跑rnn的时候会出错QAQ
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape = [None], name = "encoder{0}".format(i)))
            
        # 由于target要由decoder_inputs移动一位得到，decoder_inputs需要多接收一个变量
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape = [None], name = "decoder{0}".format(i)))
        for i in range(buckets[-1][1]):
            self.targets.append(self.decoder_inputs[i + 1])
            self.target_weights.append(tf.placeholder(tf.float32, shape = [None], name = "target{0}".format(i)))
            
        # 建立计算图，为每个bucket计算outputs, loss, 以及用于更新的op updates
        
       # self.outputs, self.losses = seq2seq_with_buckets(self.encoder_inputs, self.decoder_inputs, 
        #                                                 self.targets, self.target_weights, buckets, 
         #                                                lambda x, y: seq2seq_f(x, y, forward_only), sampled_loss)
        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, self.targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=sampled_loss)
        
        if forward_only:
            # 此时所需要的是输出，于是将输出投影到vocab_size维上
            for i in range(len(buckets)):
                for j in range(len(self.outputs[i])):
                    self.outputs[i][j] = tf.matmul(self.outputs[i][j], proj_w) + proj_b
        else:
            # 计算每个bucket对应的更新——由于每个bucket对应的loss来源不同，这里相应地有不同的updates
            params = tf.trainable_variables()
            self.updates = []
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            for i in range(len(buckets)):
                gradients = tf.gradients(self.losses[i], params)
                clipped_grad, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                # 一次updates操作会让global_step增加1
                self.updates.append(optimizer.apply_gradients(zip(clipped_grad, params), self.global_step))
            
        # 保存所有变量
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, sess, encoder_inputs, decoder_inputs, target_weights, bucket_idx, forward_only):
        '''训练/预测时，用encoder_inputs, decoder_inputs, target_weights跑一步。具体地，这里使用session.run([desired_outputs], feed_dict = {})来实现。
        Args:
            encoder_inputs, decoder_inputs, target_weights: 一组要喂的数据，都是二维的list，有序列长度个元素，每个元素是长度为batch_size的list
            bucket_idx: 这组数据对应bucket的下标
            forward_only: 若是，则用于预测，关心outputs，否则用于训练，关心updates
        Returns:
            outputs: 网络跑出的结果，是一个二维的list，有序列长度个元素，每个元素是长度为batch_size的list
            loss: 这个batch的loss
        '''
        enc_size, dec_size = self.buckets[bucket_idx]
        
        feed_dict = {}
        
        for i in range(enc_size):
            feed_dict[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in range(dec_size + 1):
            feed_dict[self.decoder_inputs[i].name] = decoder_inputs[i]
        for i in range(dec_size):
            feed_dict[self.target_weights[i].name] = target_weights[i]
            
        if forward_only:
            desired_output = []
            for i in range(dec_size):
                desired_output.append(self.outputs[bucket_idx][i])
            # desired_output.append(self.losses[bucket_idx])
            outputs = sess.run(desired_output, feed_dict)
            # 返回outputs
            return outputs
        else:
            desired_output = [self.updates[bucket_idx], self.losses[bucket_idx]]
            out_all = sess.run(desired_output, feed_dict)
            # 返回loss
            return out_all[1]

    
