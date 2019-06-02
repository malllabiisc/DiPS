from src.commons.utils import *
import numpy as np
from glob import glob


with open('data/quora_new/test_tgtX.txt', 'r', encoding='utf-8') as f:
    tgt_sents = f.read().split('\n')


all_results = {}

result_files = glob('out_quora/results_nll_submod*')
result_files.sort()
print(result_files)
for i, file in enumerate(result_files):
    hyp_txt = np.load(file)
    # try:
        # hyp_txt.shape[1]
    # except:
        # continue

    print(hyp_txt.shape)
    # print(type(hyp_txt))
    all_refs = []
    all_hyps = []
    rows = hyp_txt.shape[0]
    # cols = hyp_txt.shape[1]
    n = 0
    for j in range(rows):
        for k in range(len(hyp_txt[j])):
            all_refs.append([' '.join(tgt_sents[j].split()[:20])])
            all_hyps.append(hyp_txt[j][k])
            # print(tgt_sents[j])
            # print(hyp_txt[j][k])
            # n+=1
            # if n == 2:
                # break

        # break
    # break


    all_results[file.split("/")[-1]] = bleu_scorer(all_refs, all_hyps)

for key, value in all_results.items():
    print('{} : {}'.format(key, value))
