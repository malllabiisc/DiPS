from time import time
import pandas as pd
import numpy as np
import argparse
import pdb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os
from collections import Counter
from nltk.corpus import stopwords
from glob import glob

stops = set(stopwords.words("english"))

# python train_LSTM.py -test_file ../data/snips/test_df.csv -data_dir snips_beam -gpu 7

def train_LSTM(args):
	aug_dir = str(args.aug_dir)
	data_dir = str(args.data_dir)
	test_file= str(args.test_file)
	gpu =str(args.gpu)
	taug_dir = os.path.join(aug_dir, data_dir)
	files = glob(taug_dir+'/*' )
	print(files)
	total_files = len(files)
	print('Number of FILES: {}'.format(total_files))

	fileno=1
	for filename in files:
		print("FILE NO. : {} out of {}".format(fileno, total_files))
		fileno+=1
		print('Computing for ' + str(filename)+' ... ')
		train_file = filename.split('/')[-1]
		print('For train file: '+train_file)
		command = 'python LSTM_sentence_classifier.py -train_file {} -test_file {} -gpu {} -data_dir {}'.format(filename,test_file, gpu,data_dir)
		print('Running LSTM')
		os.system(command)

		print('Done for '+str(train_file) )

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-test_file',       type=str,               default='../data/trec/test_df.csv')
	parser.add_argument('-data_dir',       type=str,               default='trec_dbs')
	parser.add_argument('-aug_dir',       type=str,               default='../aug_out')
	parser.add_argument('-gpu',       type=int,               default=1)
	args = parser.parse_args()

	train_LSTM(args)


