from nltk import ngrams
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu







def compute_bleu(cands, ref):

    ref= ref.split()
    cands = [x.split() for x in cands]
    bleu_scores = []
    for cand in cands:
        bleu = sentence_bleu([ref], cand)
        bleu_scores.append(bleu)

    bleu_scores = np.array(bleu_scores)

    avg_bleu = bleu_scores.mean()
    max_bleu = bleu_scores.max()

    return avg_bleu, max_bleu






def avg_bleu(all_sents, refs):

    t = len(all_sents)

    bleu_mean = 0.0
    bleu_max = 0.0

    for i in range(len(all_sents)):
        ref = refs[i]
        sents = all_sents[i]
        avbleu, mbleu = compute_bleu(sents, ref)

        bleu_mean += avbleu/t
        bleu_max += mbleu/t


    return bleu_mean, bleu_max


if __name__ == '__main__':
    sent= []
    sentr = []
    sent.append('what is best way to make money online' )
    sent.append('what should i do to make money online' )
    sent.append('what should i do to earn money online' )
    sent.append('what is the easiest way to make money online' )
    sent.append('what is the easiest way to earn money online' )
    sentr.append('what s the easiest way to make money online' )
    sentr.append('what s the easiest way to earn money online' )
    sentr.append('what should i do to make money online online' )
    sentr.append('what is the best way to make money online' )
    sentr.append('what is the easiest way to make money online online' )

    all_sents = [sent]*5

    abl, mbl = avg_bleu(all_sents, sentr)
    print('Mean : {}, Max : {}'.format(abl, mbl))




