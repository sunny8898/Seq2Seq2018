#-*-coding:utf-8 -*-
# 特殊词：用于padding的<PAD>，表示未登录词的<UNK>, 表示decoder句子开始/结束的<GO>和<EOS> 
# 初步处理数据时，在target的每句话结尾加上<EOS>
# 喂数据时，再把source和target的每句话padding到对应bucket的长度.特别地，对于target中的句子，我们在前面增加<GO>，并且去掉最后一个字符

special_words = ['<PAD>', '<GO>', '<EOS>', '<UNK>'] 
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def build_vocab(data, max_vocab_size = 50000):
    '''从data创建一个词语到数字id的dict
    Args:
        data: 一个list，每个元素是包含一行中词语的list  
        max_vocab_size: 词典的大小上限。原始大小超过上限时，会丢弃出现频率最低的词语。
    Returns:
        vocab: 一个list，元素是词典中的词
        vocab_dict: 一个dict，是词到id的索引
    '''
    raw_vocab = {}
    for line in data:
        for word in line:
            if word in raw_vocab:
                raw_vocab[word] += 1
            else:
                raw_vocab[word] = 1
    vocab = special_words + sorted(raw_vocab, key = lambda x: -raw_vocab.get(x))
    print('Original Vocabulary Size is %d' % len(vocab))
    if len(vocab) > max_vocab_size:
        vocab = vocab[ : max_vocab_size]
    vocab_dict = {x : y for (x, y) in list(zip(vocab, range(len(vocab))))}
    return vocab, vocab_dict
    
def convert_to_token_ids(data, vocab, is_target = False):
    '''将data中的词语依据词典vocab转成对应的数字标号
    Args:
        data: 一个list，每个元素是包含一行中词语的list
        vocab: 用于查找词语对应标号的dict
        is_target: data是否是target，若是，将在每一行的后面加上结束的标记
    Returns:
        ids: 加上了结束标记，并将词语转成数字标号后的data
    '''
    ids = []
    for line in data:
        n_line = line
        line_ids = []
        if is_target == True:
            n_line = line + ['<EOS>']
        ids.append(list(vocab.get(w, UNK_ID) for w in n_line))
    
    return ids



def read_data(path):
    '''从path读入文件，进行预处理，返回一个list，它的每个元素是一个装着文件中某行的词语的list'''
    with open(path, "r") as f:
        lines = f.readlines()
        data = list(line.strip().split() for line in lines)
    return data


def prepare_data(trn_src_path, trn_tgt_path, max_vocab_size = 100000):
    '''返回根据training set生成的词典，以及转成token-id后的training set和development set
    Args:
        trn_src_path, trn_tgt_path, dev_src_path, dev_tgt_path: 对应文件的路径
        max_vocab_size: 生成词典的最大大小
    Returns:
        trn_src_ids, trn_tgt_ids, dev_src_ids, dev_tgt_ids: 转为token-id后的对应文件
        vocab_list: 词表
        vocab_dict: 词典
    '''
    trn_src_words = read_data(trn_src_path)
    trn_tgt_words = read_data(trn_tgt_path)
    vocab_list, vocab_dict = build_vocab(trn_src_words + trn_tgt_words, max_vocab_size)
    trn_src_ids = convert_to_token_ids(trn_src_words, vocab_dict)
    trn_tgt_ids = convert_to_token_ids(trn_tgt_words, vocab_dict, True)
    return trn_src_ids, trn_tgt_ids, vocab_list, vocab_dict

def self_test():

    raw = ["那天 ， 出去 散步 是 不 可能 了 。 其实 ， 早上 我们 还 在 光秃秃 的 灌木林 中 溜达 了 一个 小时 ， 但 从 午饭 时起 （ 无客 造访 时 ， 里德 太太 很 早就 用 午饭 ） 便 刮起 了 冬日 凛冽 的 寒风 ， 随后 阴云密布 ， 大雨滂沱 ， 室外 的 活动 也 就 只能 作罢 了 。", "那天 ， 再 出去 散步 是 不 可能 了 。 没错 ， 早上 我们 还 在 光秃秃 的 灌木林 中 漫步 了 一个 小时 ， 可是 打 从 吃 午饭 起 （ 只要 没有 客人 ， 里德 太太 总是 很早 吃 午饭 ） ， 就 刮起 了 冬日 凛冽 的 寒风 ， 随之而来 的 是 阴沉 的 乌云 和 透骨 的 冷雨 ， 这一来 ， 自然 也 就 没法 再 到 户外 去 活动 了 。", "我 倒 是 求之不得 。 我 向来 不 喜欢 远距离 散步 ， 尤其 在 冷飕飕 的 下午 。 试想 ， 阴冷 的 薄暮 时分 回得家 来 ， 手脚 都 冻僵 了 ， 还要 受到 保姆 贝茵 的 数落 ， 又 自觉 体格 不如 伊 丽莎 、 约翰 和 乔治亚 娜 ， 心里 既 难过 又 惭愧 ， 那 情形 委实 可怕 。", " 这倒 让 我 高兴 ， 我 一向 不 喜欢 远出 散步 ， 尤其 是 在 寒冷 的 下午 。 我 觉得 ， 在 阴冷 的 黄昏时分 回家 实在 可怕 ， 手指 脚趾 冻僵 了 不 说 ， 还要 挨 保姆 贝茜 的 责骂 ， 弄 得 心里 挺 不 痛快 的 。 再说 ， 自己 觉得 身体 又 比 里德 家 的 伊 丽莎 、 约翰 和 乔治 安娜 都 纤弱 ， 也 感到 低人一等 。" ] 
    n_raw = []
    for line in raw:
        n_raw.append(line.strip().split())
    vocab_list, vocab_dict = build_vocab(n_raw)


    ids = convert_to_token_ids(n_raw, vocab_dict, True)
    print(ids)
    tgt_ids = convert_to_token_ids(n_raw, vocab_dict)
    print(tgt_ids)


    # raw = read_data('clause.align.v5.train')
    # vocab_list, vocab_dict = build_vocab(raw)
    src_ids, tgt_ids, vocab_list, vocab_dict = prepare_data("./data/train.src", "./data/train.tgt", 50000)

    # ids = convert_to_token_ids(raw, vocab_dict, True)
    print(src_ids[0][ : 20])
    tmp = src_ids[0][ : 20]
    print([vocab_list[i] for i in tmp])

    print(vocab_list[: 100])
    
# self_test()
