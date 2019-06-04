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

def get_weight(count, eps=10000, min_count=2):
	if count < min_count:
		return 0.0
	else:
		return 1.00 / (count + eps)

def word_match_share(row):
	q1words = {}
	q2words = {}
	for word in str(row['question1']).lower().split():
		if word not in stops:
			q1words[word] = 1
	for word in str(row['question2']).lower().split():
		if word not in stops:
			q2words[word] = 1
	if len(q1words) == 0 or len(q2words) == 0:
		# The computer-generated chaff includes a few questions that are nothing but stopwords
		return 0
	shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
	shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
	R = (0.00+(len(shared_words_in_q1) + len(shared_words_in_q2))) /(len(q1words) + len(q2words))
	return R


def tfidf_word_match_share(row):
	q1words = {}
	q2words = {}
	for word in str(row['question1']).lower().split():
		if word not in stops:
			q1words[word] = 1
	for word in str(row['question2']).lower().split():
		if word not in stops:
			q2words[word] = 1
	if len(q1words) == 0 or len(q2words) == 0:
		# The computer-generated chaff includes a few questions that are nothing but stopwords
		return 0

	shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
	total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

	R = (np.sum(shared_weights)+0.00) / np.sum(total_weights)
	return R



def int_pred(preds, thres):
	predsx = preds.copy()
	predsx[predsx>thres] = 1
	predsx[predsx<=thres] = 0
	return predsx

np.random.seed(42)
parser = argparse.ArgumentParser(description='Paraphraser')
parser.add_argument('--model', type=str, default='none' , help='model')
args = parser.parse_args()


model = args.model
# methods =['submod', 'dbs', 'normal','dpp', 'ssr']
methods =['submod']

def log(C, x_train, y_train, x_test):
	mod=  LogisticRegression(C=C)
	mod.fit(x_train.fillna(0), y_train)
	preds =mod.predict_proba(x_test.fillna(0))
	preds = preds[:,1].copy()
	return preds

for model in methods:
	files = glob.glob('./data/merged_files/*'+ str(model)+'*' )
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

		train_df = pd.read_csv(filename)
		train_df = train_df.sample(frac= 0.5, replace=False,random_state=42)
		test_df =pd.read_csv('./data/quora/test_df.csv')

		thres = 0.4

		train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
		test_qs = pd.Series(test_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)

		train_word_match = train_df.apply(word_match_share, axis=1, raw=True)


		eps = 5000
		words = (" ".join(train_qs)).lower().split()
		counts = Counter(words)
		weights = {word: get_weight(count) for word, count in counts.items()}

		tfidf_train_word_match = train_df.apply(tfidf_word_match_share, axis=1, raw=True)


		x_train = pd.DataFrame()
		x_test = pd.DataFrame()
		x_train['word_match'] = train_word_match
		x_train['tfidf_word_match'] = tfidf_train_word_match
		x_test['word_match'] = test_df.apply(word_match_share, axis=1, raw=True)
		x_test['tfidf_word_match'] = test_df.apply(tfidf_word_match_share, axis=1, raw=True)
		Y_test= test_df['is_duplicate'].values
		y_train = train_df['is_duplicate'].values

		print('Features Computed...')
		mod=  LogisticRegression(C=4*1e-4)
		mod.fit(x_train.fillna(0), y_train)

		preds =mod.predict_proba(x_test.fillna(0))
		preds = preds[:,1].copy()
		predsx = int_pred(preds, thres)

		print('FINAL SCORES')
		print('--------------------------')

		test_score= "TEST F1 Score: {},\t TEST AUC ROC : {},\t TEST ACC: {}".format(f1_score(Y_test, predsx), roc_auc_score(Y_test, preds), accuracy_score(Y_test, predsx) )
		print(test_score)
		acc_scores =[]
		thr= 0.4
		acc_scores.append(accuracy_score(Y_test, int_pred(preds, thr)))
		sv_scores = "ACC SCORES : {}".format(str(acc_scores))
		print('--------------------------')

		fo = open('./data/scores/scores_LR_'+str(model)+ '.txt', 'a')
		fo.write(train_file+'\n')
		fo.write('-----------\n')
		fo.write(test_score+'\n')
		fo.write(sv_scores+'\n')
		fo.write('-----------\n\n\n')
		fo.close()
		print('Scores written')

	print('Done for '+str(model)+ ' with threshold: '+str(thres) )
