from src.commons.utils import *
import numpy as np
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='Bleu Evaluator')

parser.add_argument('-dataset', type=str, default='twitter', help='Which dataset to use?')
parser.add_argument('-calc_max', action="store_true",       help='Compare with maximum')

args = parser.parse_args()

if args.dataset == 'twitter':
    tgt_data_file = 'data/{}/test_tgt_new.txt'.format(args.dataset)
else:
    tgt_data_file = 'data/{}_new/test_tgtX.txt'.format(args.dataset)

with open(tgt_data_file, 'r', encoding='utf-8') as f:
    tgt_sents = f.read().split('\n')

all_tgts_refs = []
if args.dataset == 'twitter':
    for sents in tgt_sents:
        all_tgts_refs.append(sents.split('\t'))

all_results = {}
if args.dataset == 'twitter':
    result_files = glob('out_twitter_1/*.npy')
else:
    result_files = glob('out_quora_dbs/*.npy')

def get_max_bleu(hyps, refs):
    max_bleu = 0
    result = refs[0]

    for ref in refs:
        if bleu_scorer([[ref]], [hyps])[0] > max_bleu:
            result = ref
            max_bleu = bleu_scorer([[ref]], [hyps])[0]

    return result

for i, file in enumerate(result_files):
    hyp_txt = np.load(file)
    print(file)
    print(hyp_txt.shape)
    all_refs = []
    all_hyps = []
    rows = hyp_txt.shape[0]
    n = 0
    for j in range(rows):
        for k in range(len(hyp_txt[j])):
            if args.dataset == 'twitter':
                # print('------------------------')
                # print(hyp_txt[j][k])
                # print(all_tgts_refs[j][0])
                # print('------------------------')
                if args.calc_max:
                    all_refs.append([get_max_bleu(hyp_txt[j][k], all_tgts_refs[j])])
                else:
                    all_refs.append([' '.join(m.split()) for m in all_tgts_refs[j]])
            else:
                # print('------------------------')
                # print(hyp_txt[j][k])
                # print(tgt_sents[j])
                # print('------------------------')

                all_refs.append([' '.join(tgt_sents[j].split()[:20])])

            all_hyps.append(hyp_txt[j][k])
    all_results[file.split("/")[-1]] = bleu_scorer(all_refs, all_hyps)

for key, value in all_results.items():
    print('{} : {}'.format(key, value))
