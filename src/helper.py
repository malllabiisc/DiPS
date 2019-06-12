import os
import torch, pdb
import logging
from glob import glob
from torch.autograd import Variable

from nltk.translate.bleu_score import corpus_bleu
from .bleu import compute_bleu
import numpy as np

from collections import OrderedDict

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def __init__(self, expt_name):
        super(ContextFilter, self).__init__()
        self.expt_name = expt_name

    def filter(self, record):
        record.expt_name = self.expt_name
        return True

def get_logger(name, expt_name, log_format, logging_level, log_file_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.addFilter(ContextFilter(expt_name))

    return logger


def print_log(logger, dict):
    str = ''
    for key, value in dict.items():
        str += '{}: {}\t'.format(key.replace('_', ' '), value)
    str = str.strip()
    logger.info(str)


def create_save_directories(log_fname, model_fname, run_name):
    log_folder_name = os.path.join(log_fname, run_name)

    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)

    if not os.path.exists(os.path.join(model_fname, run_name)):
        os.makedirs(os.path.join(model_fname, run_name))

# Will save the checkpoint with the best validation scores (progressively), based on epoch number
def save_checkpoint(state, epoch, logger, path, args):
    sp          = path.split(".")
    file_path = '{}_{}.pth.tar'.format(sp[0].split("_")[0], epoch)
    logger.info('Saving checkpoint at : {}'.format(file_path))
    torch.save(state, file_path)

# Will load the checkpoint based on the file_path provided
def load_checkpoint(model, mode, file_path, logger, device):
    # cuda = torch.cuda.is_available()
    start_epoch = None
    train_loss = None
    val_loss = None
    voc = None
    bleu_score = None
    try:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage,
                                loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        voc = checkpoint['voc']
        bleu_score = checkpoint['bleu']
        model.to(device)
        if mode == 'decode':
            model.eval()
        else:
            model.train()
        logger.info('Successfully loaded checkpoint from {}, with epoch number: {} for {}'.format(file_path, start_epoch, mode))
        return start_epoch, train_loss, val_loss, bleu_score, voc
    except:
        logger.info('No checkpoint found on {}'.format(file_path))
        return start_epoch, train_loss, val_loss, bleu_score, voc

# Will get the latest checkpoint based on model_dir and run_name. None if no checkpoint exists in the location
def get_latest_checkpoint(model_dir, run_name, logger, epoch=None):
    dir_name    = os.path.join(model_dir, run_name)
    ckpt_names  = glob('{}/*.pth.tar'.format(dir_name))
    ckpt_names = sorted(ckpt_names)
    checkpoint = None

    if len(ckpt_names) == 0:
        logger.info('No checkpoints found in dir_name {}'.format(dir_name))
    elif epoch is not None:
        checkpoint = ckpt_names[0].replace('0', str(epoch))
    else:
        latest_eps  = max([int(k.split("/")[-1].split(".")[0].split("_")[-1]) for k in ckpt_names])
        logger.info('Checkpoint found with epoch num {}'.format(latest_eps))
        checkpoint = ckpt_names[0].replace('0', str(latest_eps))

    return checkpoint

# For pytorch
def gpu_init_pytorch(gpu_num):
    torch.cuda.set_device(int(gpu_num))
    device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")

    return device


# Bleu Scorer (Send list of list of references, and a list of hypothesis)
def bleu_scorer(ref, hyp, script='default'):
    refsend = []
    for i in range(len(ref)):
        refsi = []
        for j in range(len(ref[i])):
            refsi.append(ref[i][j].split())
        refsend.append(refsi)

    gensend = []
    for i in range(len(hyp)):
        gensend.append(hyp[i].split())

    if script == 'nltk':
         metrics = corpus_bleu(refsend, gensend)
         return [metrics]

    metrics = compute_bleu(refsend, gensend)
    return metrics


def sort_pairs_by_src_len(src, tgt):
    orig_idx = range(len(src))

    # Index by which sorting needs to be done
    sorted_idx = sorted(orig_idx, key=lambda k:len(src[k]), reverse=True)
    seq_pairs = list(zip(src, tgt))
    seq_pairs = [seq_pairs[i] for i in sorted_idx]

    # For restoring original order
    orig_idx = sorted(orig_idx, key=lambda k:sorted_idx[k])
    src, tgt = [s[0] for s in seq_pairs], [s[1] for s in seq_pairs]

    return src, tgt, orig_idx


def pad_seq(seq, max_length, voc):
    seq += [voc.w2id['EOS'] for i in range(max_length - len(seq))]
    return seq


def process_single(src, tgt, voc, device):
    src_len = len(src)
    tgt_len = len(tgt)
    src_padded = pad_seq(src, src_len, voc)
    tgt_padded = pad_seq(tgt, tgt_len, voc)

    src_var = Variable(torch.LongTensor(src_padded))
    tgt_var = Variable(torch.LongTensor(tgt_padded))

    src_var = src_var.to(device)
    tgt_var = tgt_var.to(device)

    return src_var, src_len, tgt_var, tgt_len

def process_batch(src, tgt, voc, device):
    src, tgt, orig_order    = sort_pairs_by_src_len(src, tgt)
    src_len     = [len(s) for s in src]
    src_padded  = [pad_seq(s, max(src_len), voc) for s in src]
    tgt_len     = [len(t) for t in tgt]
    tgt_padded  = [pad_seq(t, max(tgt_len), voc) for t in tgt]

    # Convert to max_len x batch_size
    src_var = Variable(torch.LongTensor(src_padded)).transpose(0, 1)
    tgt_var = Variable(torch.LongTensor(tgt_padded)).transpose(0, 1)

    src_var = src_var.to(device)
    tgt_var = tgt_var.to(device)

    return src_var, src_len, tgt_var, tgt_len, orig_order
