from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import pdb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import glob
from sklearn.linear_model import LogisticRegression
from util import ManDist, generate_confidence_intervals
import os
from collections import Counter
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

parser = argparse.ArgumentParser(description='Paraphraser')
parser.add_argument('--model', type=str, default='submod' , help='model')
parser.add_argument('--gpu', type=int, default=3, metavar='NI', help='gpu id')
args = parser.parse_args()

# models =['submod', 'dbs', 'normal','dpp', 'ssr']
model = args.model
gpu = str(args.gpu)

files = glob.glob('./data/merged_files/*'+ str(model)+'*' )
print(files)
print('Number of FILES: {}'.format(len(files)))
for file in files:
    st_time = time()
    command = 'python train.py --gpu {} --filename {}'.format( gpu, file)

    print('Running LSTM')
    os.system(command)

    et_time = time() - st_time
    print('TIME TAKEN :{}'.format(et_time/60.0))
    print('DONE FOR {}'.format(file))
