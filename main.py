import data_utils

filepaths = {
    'trn_src': './data/quora_duplicate_questions_trn.src',
    'trn_tgt': './data/quora_duplicate_questions_trn.tgt',
    'dev_src': './data/quora_duplicate_questions_dev.src',
    'dev_tgt': './data/quora_duplicate_questions_dev.tgt',
    'test_src': './data/quora_duplicate_questions_dev.src',
    'test_tgt': './data/quora_duplicate_questions_dev.tgt'
}

trn_src_ids, trn_tgt_ids, dev_src_ids, dev_tgt_ids, vocab_list = data_utils.prepare_data(filepaths['trn_src'], filepaths['trn_tgt'], filepaths['dev_src'], filepaths['dev_tgt'])

buckets = [(5, 5), (10, 10), (20, 20), (40, 40)]

encoder_inputs = [[] for _ in range(len(buckets))]
decoder_inputs = [[] for _ in range(len(buckets))]

drop_cnt = 0

for line_s, line_t in zip(trn_src_ids, trn_tgt_ids):
    for i, (size_s, size_t) in enumerate(buckets):
        if len(line_s) <= size_s and len(line_t) <= size_t:
            encoder_inputs[i].append(line_s)
            decoder_inputs[i].append(line_t)
            break
    if len(line_s) > buckets[-1][0] or len(line_t) > buckets[-1][1]:
        line_s_words = ""
        for w in line_s:
            line_s_words += " " + vocab_list[w]
        line_t_words = ""
        for w in line_t:
            line_t_words += " " + vocab_list[w]
        print('Dropped:\n%s \n %s' % (line_s_words, line_t_words))
        drop_cnt += 1

print('Sentence dropped: %d' % drop_cnt)
for i in range(len(buckets)):
    print('Bucket %d: size = (%d, %d) cnt = %d' % (i, buckets[i][0], buckets[i][1], len(encoder_inputs[i])))

            

