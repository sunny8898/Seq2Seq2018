import data_utils as du
from bleu_multiRef import bleu
res_path = "data/test.res"
ans_path = "data/test.ans"

#ref_num = 3
ref_num = 1

res_ids, raw_ans_ids, vocab_list, vocab_dict = du.prepare_data(res_path, ans_path)


ans_ids = []
ans_lens = []
for line, i in zip(raw_ans_ids, range(len(raw_ans_ids))):
    if i % ref_num == 0:
        ans_ids.append([line[: -1]])
        ans_lens.append([len(line) - 1])
    else:
        ans_ids[-1].append(line[: -1])
        ans_lens[-1].append(len(line) - 1)

print(bleu(res_ids, ans_ids, ans_lens))
