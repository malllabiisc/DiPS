import os, re
import torch
import unicodedata

from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")





class ParaphraseDataset(Dataset):
    def __init__(self, dataset, datatype, max_length=20, is_debug=False, is_train=False):
        orig_file = os.path.join('data',dataset, datatype, 'src.txt')
        para_file = os.path.join('data',dataset, datatype, 'tgt.txt')

        with open(orig_file, 'r', encoding='utf-8', errors='ignore') as f:
            self.orig_sents = f.read().split('\n')[:-1]
        with open(para_file, 'r', encoding='utf-8', errors='ignore') as f:
            self.para_sents = f.read().split('\n')[:-1]

        if is_debug:
            self.orig_sents = self.orig_sents[:5000]
            self.para_sents = self.para_sents[:5000]

        all_sents = zip(self.orig_sents, self.para_sents)

        if is_train:
            all_sents = sorted(all_sents, key= lambda h: len(h[0].split()))
        self.orig_sents, self.para_sents = zip(*all_sents)

        # print(self.orig_sents[:10], self.para_sents[:10])

        if is_train:
            self.orig_sents, self.para_sents = self.orig_sents + self.para_sents, self.para_sents + self.orig_sents

        self.max_length = max_length

    def __len__(self):
        return len(self.orig_sents)

    def __getitem__(self, idx):
        src  = self.process_string(self.unicodeToAscii(self.orig_sents[idx]))
        tgt  = self.process_string(self.unicodeToAscii(self.para_sents[idx]))
        pair = {'src': self.curb_to_length(src), 'tgt': self.curb_to_length(tgt)}
        return pair

    def curb_to_length(self, string):
        return ' '.join(string.strip().split()[:self.max_length])

    def process_string(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " 's", string)
        string = re.sub(r"\'ve", " 've", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " 're", string)
        string = re.sub(r"\'d", " 'd", string)
        string = re.sub(r"\'ll", " 'll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string

    def unicodeToAscii(self, string):
        return ''.join(c for c in unicodedata.normalize('NFD', string)
                       if unicodedata.category(c) != 'Mn')





class Voc:
    def __init__(self, name):
        self.name       = name
        self.trimmed    = False
        self.frequented = False
        self.w2id       = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.id2w       = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.w2c        = {}
        self.nwords     = 3

    def addSentence(self, sent):
        for word in sent.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.w2id:
            self.w2id[word]     = self.nwords
            self.id2w[self.nwords]   = word
            self.w2c[word]      = 1
            self.nwords         = self.nwords + 1
        else:
            self.w2c[word]      = self.w2c[word] + 1

    def trim(self, mincount):
        if self.trimmed == True:
            return
        self.trimmed    = True

        keep_words = []
        for k, v in self.w2c.items():
            if v >= mincount:
                keep_words += [k]*v

        self.w2id       = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.id2w       = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.w2c        = {}
        self.nwords     = 3
        for word in keep_words:
            self.addWord(word)

    def most_frequent(self, topk):
        if self.frequented == True:
            return
        self.frequented     = True

        keep_words = []
        count      = 3
        sorted_by_value = sorted(self.w2c.items(), key=lambda kv: kv[1], reverse=True)
        for word, freq in sorted_by_value:
            keep_words  += [word]*freq
            count += 1
            if count == topk:
                break

        self.w2id       = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.id2w       = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.w2c        = {}
        self.nwords     = 3
        for word in keep_words:
            self.addWord(word)




def indicesFromSentence(voc, sent, max_length):
    idx_vec = []
    for w in sent.split(' '):
        try:
            idx = voc.w2id[w]
            idx_vec.append(idx)
        except:
            idx_vec.append(voc.w2id['UNK'])
    # idx_vec.append(voc.w2id['EOS'])
    if len(idx_vec) < max_length-1:
        idx_vec.append(voc.w2id['EOS'])
    return idx_vec

def indicesFromSentences(voc, sents, max_length):
    all_indexes = []
    for sent in sents:
        all_indexes.append(indicesFromSentence(voc, sent, max_length))
    return all_indexes





def tensorFromSentence(voc, sentence, device, max_length):
    indexes = indicesFromSentence(voc, sentence, max_length)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(voc, src, tgt, device, max_length):
    src_tens = tensorFromSentence(voc, src, device, max_length)
    tgt_tens = tensorFromSentence(voc, tgt, device, max_length)
    return src_tens, tgt_tens

def tensorsFromPairs(voc, src, tgt, device, max_length):
    batch_src_tensors = []
    batch_tgt_tensors = []
    for s, t in zip(src, tgt):
        src_tens, tgt_tens = tensorsFromPair(voc, s, t, device, max_length)
        batch_src_tensors.append(src_tens)
        batch_tgt_tensors.append(tgt_tens)
    return batch_src_tensors, batch_tgt_tensors





def indicesToSentence(voc, tensor, no_eos=False):
    sentence_word_list =  []
    for idx in tensor:
        w = voc.id2w[idx.item()]
        if no_eos:
            if w != 'EOS':
                sentence_word_list.append(w)
        else:
            sentence_word_list.append(w)
    return sentence_word_list


def indicesToSentences(voc, tensors, no_eos=False):
    tensors = tensors.transpose(0, 1)
    batch_word_list = []
    for tens in tensors:
       batch_word_list.append(indicesToSentence(voc, tens, no_eos))
    return batch_word_list
