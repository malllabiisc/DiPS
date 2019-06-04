# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import os, sys
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
import argparse

dropout = 0.8  #float(sys.argv[1])

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, device):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.drop_hidden = nn.Dropout(p=dropout)
        self.hidden = self.init_hidden(device=device)


    def init_hidden(self ,device):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).to(device),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).to(device))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        y = self.drop_hidden(y)
        log_probs = F.log_softmax(y)
        return log_probs

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train(train_file, test_file, data_dir, device):
    # train_data, dev_data, test_data, word_to_ix, label_to_ix = data_loader.load_MR_data()
    train_data, dev_data, test_data, word_to_ix, label_to_ix = data_loader.load_trec_data(train_file, test_file)

    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 20
    best_test_acc = 0.0
    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix) ,device = device )
    model =model.to(device)
    loss_function = nn.NLLLoss()
    loss_function.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    no_up = 0
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    all_test_acc= []
    for i in range(EPOCH):
        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        model.train()
        train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i, device)
        model.eval()
        print('now best test acc:', best_test_acc, 'drop', dropout)
        # dev_acc = evaluate(model,dev_data,loss_function,word_to_ix,label_to_ix,'dev')
        test_acc = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix,device,'test')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            all_test_acc.append(test_acc)
        #     os.system('rm mr_best_model_acc_*.model')
        #     print('New Best Dev!!!')
        # torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(test_acc*10000)) + '.model')
        #     no_up = 0
        # else:
        #     no_up += 1
        #     if no_up >= 10:
        #         exit()

    save_dir =  './val/LSTM_scores_'+ str(data_dir)+'.txt'
    test_score= "TEST ACC: {}".format(best_test_acc)
    sv_scores = "M ACC SCORES : {}".format(str(all_test_acc))
    print(test_score)
    fo = open(save_dir, 'a')

    fo.write(train_file+'\n')
    fo.write('-----------\n')
    fo.write(test_score+'\n')
    fo.write(sv_scores+'\n')
    # fo.write(cid+'\n')
    fo.write('-----------\n\n\n')
    fo.close()
    print('Scores written')

    print('Done for '+str(train_file) )

def evaluate(model, data, loss_function, word_to_ix, label_to_ix, device, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden(device)
        sent = data_loader.prepare_sequence(sent, word_to_ix).to(device)
        label = data_loader.prepare_label(label, label_to_ix).to(device)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.data.item()
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc



def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i, device):
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:

        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden(device)
        sent = data_loader.prepare_sequence(sent, word_to_ix)
        sent =sent.to(device)
        label = data_loader.prepare_label(label, label_to_ix).to(device)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data.item()
        count += 1
        # if count % 500 == 0:
        #     print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data.item()))

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM Classifier')
    parser.add_argument('-train_file',       type=str,               default='../data/trec/train_df.csv')
    parser.add_argument('-test_file',       type=str,               default='../data/trec/test_df.csv')
    parser.add_argument('-data_dir',       type=str,               default='trec_dbs')
    parser.add_argument('-gpu',       type=str,               default='7')
    parser.add_argument('-out_file',       type=str,               default='data/')
    args = parser.parse_args()


    train_file = args.train_file
    test_file = args.test_file
    gpu = args.gpu
    data_dir = str(args.data_dir)
    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")



    train(train_file, test_file, data_dir, device)
