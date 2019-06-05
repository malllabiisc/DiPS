import numpy as np
import scipy.linalg as la
import pdb
from nltk import ngrams
import difflib
import pickle
from time import time
import os

from scipy.spatial.distance import pdist, squareform
import scipy
from numpy import dot
from numpy.linalg import norm

#####################################################################################################################

########################################### RBF QUALITY FUNCTION ####################################################

embedding_path = os.path.join('data', 'embeddings', 'word2vec.pickle')
model = None

if os.path.exists(embedding_path):
    print('Loading Word2Vec')
    st_time = time()

    with open(embedding_path, 'rb') as f:
        model = pickle.load(f)

    print('Word2vec Loaded')
    etime = (time() - st_time)/60.0
    print('Time Taken : {}'.format(etime))
    
cos_sim = lambda a,b: dot(a, b)/(norm(a)*norm(b))
rbf = lambda a,b, sigma : scipy.exp(-(np.sum( (a-b)**2 ) )/ sigma ** 2)

def sent2wvec(s):
    v= []
    for w in s:
        try:
            vec =  model[w]
            v.append(vec)
        except:
            vec = np.random.random(300)
            v.append(vec)

    v = np.array(v)
    return v

def sentence_compare(s1, s2, kernel='cos', **kwargs):
    l1 = s1.split()
    l2 = s2.split()

    v1= sent2wvec(l1)
    v2= sent2wvec(l2)
    # v2 = np.array([model.wv.word_vec(w) for w in l2])
    score = 0
    len_s1 = v1.shape[0]
    for v in v1:
        if kernel == 'cos':
            wscore = np.max(np.array([cos_sim(v,i) for i in v2] ))
        elif kernel == 'rbf':
            wscore = np.max(np.array([rbf(v,i, kwargs['sigma']) for i in v2] ))
        else:
            print('Error in kernel type')
        score += wscore/len_s1

    return score


def similarity_func(v, S):
    if len(S):
        score = 0.0

        for sent in S:
            score+= sentence_compare(v, sent, kernel='rbf', sigma=1.0)

        return np.sqrt(score)
    else:
        return 0.0


def similarity_gain(v, s, base_score=0.0):
    score = 0.0
    score+= sentence_compare(v, s, sigma=1.0)
    score += (base_score**2)

    return np.sqrt(score)


#####################################################################################################################
#####################################################################################################################

########################################### NGRAM FUNCTIONS #########################################################



def ngram_toks(sents, n=1):
    ntoks =[]
    for sent in sents:
        ntok = list(ngrams(sent.split(), n))
        newtoks = [tok for tok in ntok]
        ntoks+= newtoks
    return ntoks

def distinct_ngrams(S):
    if len(S):
        S = ' '.join(S)
        N = [1,2,3]
        score = 0.0
        for n in N:
            toks = set(ngram_toks([S], n))
            score+= (1.0/n) * len(toks)

        return score
    else:
        return 0.0

def ngram_overlap(v, S):
    if len(S):
        N = [1,2,3]
        score = 0.0


        for n in N:
            src_toks = set(ngram_toks([v], n))
            for sent in S:
                sent_toks = set(ngram_toks(S, n))

                overlap = src_toks.intersection(sent_toks)

                score += (1.0/(4-n)) * len(overlap)

        return np.sqrt(score)
    else:
        return 0.0

def ngram_overlap_unit(v, S , base_score=0.0):
    N = [1,2,3]
    score = 0.0
    try:
        temp = S[0]
    except:
        S = [S]

    for n in N:
        src_toks = set(ngram_toks([v], n))
        sent_toks = set(ngram_toks([S], n))
        overlap = src_toks.intersection(sent_toks)

        score += (1.0/(4-n)) * len(overlap)

    return np.sqrt( (base_score**2) + score)


#####################################################################################################################

########################################### EDIT DISTANCE FUNCTION ##################################################

def seq_func(V, S):
    if len(S):
        score = 0.0
        for v in V:
            for s in S:
                vx = v.split()
                sx = s.split()

                seq = difflib.SequenceMatcher(None, vx, sx)
                score+= seq.ratio()

        return np.sqrt(score)
    else:
        return 0.0

def seq_gain(V, s, base_score= 0.0):
    gain =0.0
    for v in V:
        vx = v.split()
        sx = s.split()

        seq = difflib.SequenceMatcher(None, vx, sx)
        gain+= seq.ratio()

    score =(base_score**2) + gain

    return np.sqrt(score)

def info_func(S, orig_count, ref_count):
    if len(S):
        score =0.0
        for s in S:
            stoks = set(s.split())
            orig_toks = set(orig_count.keys())

            int_toks = stoks.intersection(orig_toks)
            for tok in int_toks:
                try:
                    score+= (orig_count[tok] / (1 + ref_count[tok] ))
                except:
                    score+= orig_count[tok]

        return np.sqrt(score)

    else:
        return 0.0

def info_gain(s, orig_count, ref_count, base_score =0.0):
    score =0.0
    stoks = set(s.split())
    orig_toks = set(orig_count.keys())

    int_toks = stoks.intersection(orig_toks)
    for tok in int_toks:
        try:
            score+= (orig_count[tok] / (1 + ref_count[tok] ))
        except:
            score+= orig_count[tok]

    score+= (base_score**2)

    return np.sqrt(score)
