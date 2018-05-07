import data_utils as du
from bleu_multiRef import bleu
res_path = "data/test.res"
ans_path = "data/test.ans"

res_ids, ans_ids, vocab_list, vocab_dict = du.prepare_data(res_path, ans_path)

print(bleu(res_ids, [[x[: -1]] for x in ans_ids], [[len(line)] for line in ans_ids]))
