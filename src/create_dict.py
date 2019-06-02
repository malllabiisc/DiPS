import numpy as np
import os, math, logging, pdb
from attrdict import AttrDict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
from time import time
from gensim import models
import argparse

from src.model import s2s
from src.helper import *

def main(args):
    log_folder_name = os.path.join('Logs', args.run_name)
    logger = get_logger(__name__, args.run_name, args.log_fmt, logging.INFO, os.path.join(log_folder_name, 'vec_loader.log'))
    checkpoint = get_latest_checkpoint('Model', args.run_name, logger)
    device = gpu_init_pytorch(args.gpu)

    logger.info('Loading Vocabulary file')

    with open(os.path.join('Model', args.run_name, 'vocab.p'), 'rb') as f:
        voc = pickle.load(f)

    logger.info('Vocabulary file Loaded from {}'.format(os.path.join('Model', args.run_name, 'vocab.p')))


    # Load models
    # Load Dictionary
    logger.info('Loading model')
    config_file_name = os.path.join('Model', args.run_name, 'config.p')
    with open(config_file_name, 'rb') as f:
        arg_prev = AttrDict(pickle.load(f))
    model = s2s(arg_prev, voc, device, logger)
    _, _, _, _, voc = load_checkpoint(model, 'decode', checkpoint, logger, device)

    worddict = {}
    k = 0

    logger.info('Create dictionary using {} embeddings'.format(args.model))
    if args.model == 'pretrained':
        pretrained_path = os.path.join('data', 'GoogleNews-vectors-negative300.bin.gz')
        logger.info('Loading pretrained dictionary from {}'.format(pretrained_path))
        st = time()
        w2v = models.KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        et = (time() - st)/60.0
        logger.info('Loaded pretrained dictionary from {} in {}'.format(pretrained_path, et))
        worddict = {}
        for word, _ in voc.w2id.items():
            try:
                worddict[word] = w2v[word]
            except:
                worddict[word] = np.random.randn(300)
                k+=1

    else:
        embed = model.embedding
        for word, idx in voc.w2id.items():
            try:
                worddict[word] = embed(torch.LongTensor([idx]).to(device)).cpu()[0].detach().numpy()
            except:
                k += 1
                worddict[word] = np.random.randn(arg_prev.emb_size)

    logger.info('{} words not found in {}'.format(k, args.model))

    word2vec_path = os.path.join('data', 'embeddings', 'word2vec.pickle')
    if os.path.exists(word2vec_path):
        logger.info('File {} already present. Overwriting existing file'.format(word2vec_path))
    else:
        logger.info('Saving dictionary at {}'.format(word2vec_path))

    with open(word2vec_path, 'wb') as f:
        pickle.dump(worddict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser to build dictionary for submodular function run')

    parser.add_argument('-model', type=str, default='trained', choices=['trained', 'pretrained'], help='Create dictionary of these word embeddings')
    parser.add_argument('-gpu', type=str, required=True, help='Specify the gpu to use')
    parser.add_argument('-run_name', type=str, required=True, help='Run name from which the model, and dictionary needs to be loaded')
    parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

    # Dont change ckpt_file
    parser.add_argument('-ckpt_file', type=str, default='s2s_0.pth.tar', help='Checkpoint file name')

    args = parser.parse_args()
    main(args)

