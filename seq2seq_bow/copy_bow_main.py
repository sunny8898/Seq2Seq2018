#coding:utf-8
import random
import numpy as np
import tensorflow as tf
import time
import sys
import data_utils as du
from copy_bow_model import Seq2SeqModel

buckets = [(10 * i - 9, 10 * i) for i in range(1, 25)]

vocab_size = 450
batch_size = 32
rnn_size = 32
attention_depth = rnn_size
encoder_layers = 3
decoder_layers = 2
max_gradient_norm = 10

max_lbd = 1
lbd_alpha = 0.1
lbd_beta = 0.1

checkpoint_freq = 100
verbose_freq = 20

train_path = './train/'


def list_transpose(inputs):
        '''将inputs代表的矩阵进行转置'''
        a = np.array(inputs)
        b = np.transpose(inputs)
        return list(list(c) for c in b)

    
def put_data_into_buckets(source, target):
    '''将source,target中的句对放入长度合适的bucket里
    Args:
        source, target: 两个list，每个元素是包含source/target中一个句子中词语的list
    Returns:
        encoder_inputs, decoder_inputs: 两个list，第i个元素是包含了长度落在第i个bucket中的句子的list
    '''

    encoder_inputs = [[] for _ in range(len(buckets))]
    decoder_inputs = [[] for _ in range(len(buckets))]

    drop_cnt = 0

    for line_s, line_t in zip(source, target):
        for i, (size_s, size_t) in enumerate(buckets):
            if len(line_s) < size_s and len(line_t) < size_t:
                encoder_inputs[i].append(line_s)
                decoder_inputs[i].append(line_t)
                break
        if len(line_s) >= buckets[-1][0] or len(line_t) >= buckets[-1][1]:
            drop_cnt += 1

    print('Sentence dropped: %d' % drop_cnt)
    for i in range(len(buckets)):
        print('Bucket %d: size = (%d, %d) cnt = %d' % (i, buckets[i][0], buckets[i][1], len(encoder_inputs[i])))

    return encoder_inputs, decoder_inputs

def get_copy_dict(vocab):
    lines = du.read_data(filepaths['copypp'])
    ids = du.convert_to_token_ids(lines, vocab)
    copy_dict = {}
    for line in ids:
        copy_dict[line[0]] = line[1: ]
    return copy_dict



filepaths = {
'trn_src': './data/train.src',
'trn_tgt': './data/train.tgt',
'copypp': './data/copypp.pp'
}
trn_src_ids, trn_tgt_ids, vocab_list, vocab_dict = du.prepare_data(filepaths['trn_src'], filepaths['trn_tgt'], vocab_size)
trn_encoder_inputs, trn_decoder_inputs = put_data_into_buckets(trn_src_ids, trn_tgt_ids)
copy_dict = get_copy_dict(vocab_dict)


def get_batch(encoder_inputs_all, decoder_inputs_all, bucket_idx, batch_size):
    '''从encoder_inputs_all, decoder_inputs_all的第bucket_idx个bucket里，随机选出batch_size大小的句对
    Args:
        encoder_inputs_all, decoder_inputs_all: 两个list，第i个元素是包含了所有长度落在第i个bucket中的句子的list
        bucket_idx: 要从中取句对的bucket的编号
        batch_size: batch的大小
    Returns:
        encoder_inputs, decoder_inputs: [bucket_size(+1) *
        batch_size]两个分别包含了对应bucket_size/bucket_size+1个元素的list，第i个元素是一个长度为batch_size的list，代表每个句子第i个词
        target_weights: [bucket_size * batch_size]用于决定计算loss时是否考虑结果的某个位：若target的该位是padding出来的，就置为0，否则置为1
        
    '''
    bucket_len = len(encoder_inputs_all[bucket_idx])
    enc_size, dec_size = buckets[bucket_idx]
    enc_lines = []
    dec_lines = []
    tgt_lines = []
    for _ in range(batch_size):
        idx = random.randint(0, bucket_len - 1)
        enc_line = encoder_inputs_all[bucket_idx][idx]
        dec_line = decoder_inputs_all[bucket_idx][idx]
        # encoder_input padding到对应长度即可
        enc_line = enc_line + [du.PAD_ID] * (enc_size - len(enc_line)) 
        # decoder_input padding到对应长度-1，再在前面加上<GO>
        # decoder_target padding到对应长度
        tgt_line = dec_line + [du.PAD_ID] * (dec_size - len(dec_line))
        dec_line = [du.GO_ID] + dec_line + [du.PAD_ID] * (dec_size - len(dec_line) - 1) 
        enc_lines.append(enc_line)
        dec_lines.append(dec_line)
        tgt_lines.append(tgt_line)
    
    # 转置, [T, B]
    encoder_inputs = list_transpose(enc_lines)
    decoder_inputs = list_transpose(dec_lines)
    decoder_targets = list_transpose(tgt_lines)
    target_weights = []
    for len_idx in range(dec_size):
        target_weight = [1] * batch_size
        for batch_idx in range(batch_size):
            # 第batch_idx个句子的target的第len_idx位，对应了decoder_inputs的第len_idx + 1 位
            #  h   e  l  l  o  <EOS>  <-- target
            # []   [] [] [] []  []    <-- rnn cells
            # <GO> h  e  l  l   o     <-- decoder_inputs
            # 若这一位是<PAD>, 计算loss时它不计入，将weight置为0
            if (decoder_targets[len_idx][batch_idx] == du.PAD_ID):
                target_weight[batch_idx] = 0
        target_weights.append(target_weight)
    return encoder_inputs, decoder_inputs, decoder_targets, target_weights

    


def load_model(sess):
    model = Seq2SeqModel(vocab_size,
                         rnn_size,
                         encoder_layers,
                         decoder_layers,
                         attention_depth,
                         max_gradient_norm,
                         copy_dict)
    checkpoint = tf.train.get_checkpoint_state(train_path)
    if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path):
        model.saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Load model parameters from %s." % train_path)
    else:
        sess.run(tf.global_variables_initializer()) # ??
        print("Create a new model.")
    return model

def train():
    
    # 训练时，先随机一个bucket，再从其中随机一个batch，这里先计算每个bucket所含句子的个数，从而在之后使得每个bucket被随机到的概率与包含元素个数成比例
    trn_total = 0 # 训练集总大小
    trn_bucket_lens = [] # 每个bucket里训练集的大小列表
    for i in range(len(buckets)):
        trn_bucket_lens.append(len(trn_encoder_inputs[i]))
    trn_total = sum(trn_bucket_lens)
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        model = load_model(sess)
        file_writer = tf.summary.FileWriter(".", sess.graph)
        
        print("Model successfully loaded.")
        
        current_time_step = 0
        average_loss, average_rnn_loss, average_bow_loss, average_time = 0.0, 0.0, 0.0, 0.0
        
        while (True):
            current_time_step += 1
            
            # 随机选择一个bucket，从中取出一个batch训练模型
            rand = random.randint(1, trn_total) 
            bucket_idx = -1
            for idx in range(len(buckets)):
                rand -= trn_bucket_lens[idx]
                if rand <= 0:
                    bucket_idx = idx
                    break
        
            current_time_start = time.time()
            encoder_inputs, decoder_inputs, decoder_targets, target_weights = get_batch(
                trn_encoder_inputs, 
                trn_decoder_inputs, 
                bucket_idx, batch_size)

            epoch = int(model.get_cnt(sess)) // trn_total
            epoch = max(0, epoch - 10)

            lbd = min(lbd_alpha + lbd_beta * epoch, max_lbd)

            cur_loss, cur_rnn_loss, cur_bow_loss, summaries = model.step(
                sess, encoder_inputs, decoder_inputs, decoder_targets,
                target_weights, lbd)

            file_writer.add_summary(summary = summaries, global_step =
                                   sess.run(model.global_step))
            
            average_loss += cur_loss / verbose_freq
            average_rnn_loss += cur_rnn_loss / verbose_freq
            average_bow_loss += cur_bow_loss / verbose_freq
            average_time += (time.time() - current_time_start) / verbose_freq
            
            if current_time_step % checkpoint_freq == 0:
                # 保存当前参数
                checkpoint_path = train_path + "seq2seq.ckpt"
                model.saver.save(sess, checkpoint_path, global_step = model.global_step)
                print("saving model global step = %d" % model.global_step.eval())
            if current_time_step % verbose_freq == 0:
                # 打印这个阶段的信息
                print("global step = %d, lbd = %lf, average loss = %.3lf, rnn_loss = %.3lf, bow_loss = %.3lf, average time = %.3lf" % 
                      (model.global_step.eval(), lbd, average_loss,
                       average_rnn_loss, average_bow_loss, average_time))
                average_loss, average_rnn_loss, average_bow_loss, average_time = 0.0, 0.0, 0.0, 0.0
                
        
                
def predict():
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = load_model(sess)
        
        while True:
            sys.stdout.write("Please input a sentence...\n")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            sentence = sentence.strip().split()

            # [B = 1, T]
            ids = du.convert_to_token_ids([sentence], vocab_dict)
            encoder_inputs = list_transpose(ids)

            # outputs是一个包含bucket[bucket[idx]][1]个元素的list，每个元素是一个batch_size * vocab_size的np.array
            predict_ids = model.predict(sess, encoder_inputs)
            predict_words = [vocab_list[id[0]] for id in predict_ids[0].tolist()]
            # 截断<EOS>以后的部分
            truncate = len(predict_words)
            for i in range(truncate):
                if predict_words[i] == '<EOS>':
                    truncate = i
                    break
            predict_sentence = ' '.join(predict_words[: truncate])
            print('Predicted paraphrase: %s' % predict_sentence)

def predict_a_lot():
    tf.reset_default_graph()

    with tf.Session() as sess:
        model = load_model(sess)
        
        while True:
            sentence = input()
            if len(sentence.strip().split()) == 0:
                continue

            # print(sentence)
            sentence = sentence.strip().split()

            # [B = 1, T]
            ids = du.convert_to_token_ids([sentence], vocab_dict)
            encoder_inputs = list_transpose(ids)

            # outputs是一个包含bucket[bucket[idx]][1]个元素的list，每个元素是一个batch_size * vocab_size的np.array
            predict_ids = model.predict(sess, encoder_inputs)
            predict_words = [vocab_list[id[0]] for id in predict_ids[0].tolist()]
            # 截断<EOS>以后的部分
            truncate = len(predict_words)
            for i in range(truncate):
                if predict_words[i] == '<EOS>':
                    truncate = i
                    break
            predict_sentence = ' '.join(predict_words[: truncate])
            print(predict_sentence)
            # print("------------------------------------------------")


train()


