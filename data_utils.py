
# 特殊词：用于padding的<PAD>，表示未登录词的<UNK>, 表示decoder句子开始/结束的<GO>和<EOS> 
# 初步处理数据时，在target的每句话结尾加上<EOS>
# 喂数据时，再把source和target的每句话padding到对应bucket的长度.特别地，对于target中的句子，我们在前面增加<GO>，并且去掉最后一个字符

special_words = ['<PAD>', '<GO>', '<EOS>', '<UNK>'] 
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def build_vocab(data, max_vocab_size = 100000):
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


def prepare_data(trn_src_path, trn_tgt_path, dev_src_path, dev_tgt_path, max_vocab_size = 100000):
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
    dev_src_words = read_data(dev_src_path)
    dev_tgt_words = read_data(dev_tgt_path)
    vocab_list, vocab_dict = build_vocab(trn_src_words + trn_tgt_words, max_vocab_size)
    trn_src_ids = convert_to_token_ids(trn_src_words, vocab_dict)
    trn_tgt_ids = convert_to_token_ids(trn_tgt_words, vocab_dict, True)
    dev_src_ids = convert_to_token_ids(dev_src_words, vocab_dict)
    dev_tgt_ids = convert_to_token_ids(dev_tgt_words, vocab_dict, True)
    return trn_src_ids, trn_tgt_ids, dev_src_ids, dev_tgt_ids, vocab_list, vocab_dict

def self_test():

    raw = ["why are people on this site so obsessed with iq",
    "how do capitalism and communism differ from each other",
    "who are the celebrity who are on quora",
    "how can i burn my fat",
    "what do indian think of donald trump"]

    n_raw = []
    for line in raw:
        n_raw.append(line.strip().split())
    vocab_list, vocab_dict = build_vocab(n_raw)


    ids = convert_to_token_ids(n_raw, vocab_dict, True)
    print(ids)
    tgt_ids = convert_to_token_ids(n_raw, vocab_dict)
    print(tgt_ids)


    raw = read_data('./data/quora_duplicate_questions_trn.src')
    raw_tgt = read_data('./data/quora_duplicate_questions_trn.tgt')
    vocab_list, vocab_dict = build_vocab(raw_tgt + raw)

    ids = convert_to_token_ids(raw, vocab_dict, True)
    print(ids[: 10])
    tgt_ids = convert_to_token_ids(raw_tgt, vocab_dict)
    print(tgt_ids[: 10])

    print(vocab_list[: 100])
    
# self_test()
