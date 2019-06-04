from time import time
import pandas as pd
import numpy as np
import argparse
import pdb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import os
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from glob import glob

stops = set(stopwords.words("english"))

def int_pred(preds, thres):
	predsx = preds.copy()
	predsx[predsx>thres] = 1
	predsx[predsx<=thres] = 0
	return predsx

np.random.seed(42)
def log(C, x_train, y_train, x_test):
	mod=  LogisticRegression(C=C,solver='lbfgs')
	mod.fit(x_train, y_train)

	preds =mod.predict(x_test)
	return preds

def train_LR(args):

	aug_dir = str(args.aug_dir)
	data_dir = str(args.data_dir)
	test_file= str(args.test_file)
	taug_dir = os.path.join(aug_dir, data_dir)
	test_df =pd.read_csv(test_file)
	test_df = test_df[['text', 'label']]

	files = glob(taug_dir+'/*' )
	print(files)
	total_files = len(files)
	print('Number of FILES: {}'.format(total_files))
	multi_class= True

	fileno=1
	for filename in files:
		print("FILE NO. : {} out of {}".format(fileno, total_files))
		fileno+=1
		print('Computing for ' + str(filename)+' ... ')
		train_file = filename.split('/')[-1]
		print('For train file: '+train_file)
		train_df = pd.read_csv(filename)
		train_df['text'] = train_df['text'].apply(lambda x: x.replace('<eos>', ''))
		train_df['text'] = train_df['text'].apply(lambda x: x.replace('<unk>', ''))
		print(train_df.shape)

		ftrain_df = train_df.copy()
		full_df = ftrain_df.append(test_df)

		tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
		features = tfidf.fit_transform(full_df['text']).toarray()
		labels = full_df['label']
		if len(set(labels))>2:
			multi_class = True
		else:
			multi_class = False
		print(features.shape)

		X_train = features[:ftrain_df.shape[0]]
		y_train = labels[:ftrain_df.shape[0]]

		X_test = features[ftrain_df.shape[0]:]
		y_test = labels[ftrain_df.shape[0]:]

		print('Features Computed...')
		if multi_class:
			clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
		else:
			clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

		acc_score = clf.score(X_test, y_test)
		print('Score for file {} samples: {}'.format(filename, acc_score ))
		preds = clf.predict(X_test)

		print('FINAL SCORES')
		print('--------------------------')
		test_score= "TEST ACC: {}".format( accuracy_score(y_test, preds) )
		print(test_score)
		acc_scores =[]
		acc_scores.append(accuracy_score(y_test, log(1.1, X_train, y_train, X_test)))

		sv_scores = "M ACC SCORES : {}".format(str(acc_scores))
		print('--------------------------')
		save_dir =  './scores/LR_scores_'+ str(data_dir)+'.txt'
		fo = open(save_dir, 'a')
		fo.write(train_file+'\n')
		fo.write('-----------\n')
		fo.write(test_score+'\n')
		fo.write(sv_scores+'\n')
		fo.write('-----------\n\n\n')
		fo.close()
		print('Scores written')

		print('Done for '+str(train_file) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LogReg Classifier')
    parser.add_argument('-test_file',       type=str,               default='./data/trec/test_df.csv')
    parser.add_argument('-data_dir',       type=str,               default='trec_dbs')
    parser.add_argument('-aug_dir',       type=str,               default='aug_out')
    args = parser.parse_args()




    train_LR(args)

