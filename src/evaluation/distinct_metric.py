from nltk import ngrams


def ngram_toks(sents, n=1):
    ntoks =[]
    for sent in sents:
        ntok = list(ngrams(sent.split(), n))
        newtoks = [tok for tok in ntok]
        ntoks+= newtoks
    return ntoks



def ndistinct(sents, n=1):
    total_tokens = ngram_toks(sents, n)

    unique_toks = set(total_tokens)
    tlen = len(total_tokens)
    ulen = float(len(unique_toks))
    return ulen/tlen

def avg_nd(all_sents, n=1):
    scores=0
    t = len(all_sents)
    for sents in all_sents:
        scores+= (ndistinct(sents,n) / t)

    return scores


if __name__ == '__main__':
    sent= []
    sent.append('what is best way to make money online' )
    sent.append('what should i do to make money online' )
    sent.append('what should i do to earn money online' )
    sent.append('what is the easiest way to make money online' )
    sent.append('what is the easiest way to earn money online' )
    sent.append('what s the easiest way to make money online' )
    sent.append('what s the easiest way to earn money online' )
    sent.append('what should i do to make money online online' )
    sent.append('what is the best way to make money online' )
    sent.append('what is the easiest way to make money online online' )

    for i in range(1,8):
        print('Number of {}-distinct ngrams: {}'.format(i, ndistinct(sent,i)))

