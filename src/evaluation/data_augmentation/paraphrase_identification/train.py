from time import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist, generate_confidence_intervals
import os
from keras.backend.tensorflow_backend import set_session
np.random.seed(42)

parser = argparse.ArgumentParser(description='Paraphraser')
parser.add_argument('--gpu', type=int, default=3, metavar='NI', help='gpu id')
parser.add_argument('--filename', type=str, default='train_df.csv' , help='Train File')
args = parser.parse_args()

filename = args.filename
gpuid = args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

BUILD_EMBED =True
# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20

def int_pred(preds, thres):
	predsx = preds.copy()
	predsx[predsx>thres] = 1
	predsx[predsx<=thres] = 0
	return predsx

# File paths
TRAIN_CSV = str(filename)
train_file= TRAIN_CSV.split('/')[-1]
print(train_file)
TEST_CSV= './data/quora/test_df.csv'
# Load training set
train_df = pd.read_csv(TRAIN_CSV)
train_size = train_df.shape[0]
# train_df = train_df.iloc[:int(train_size/5.0)]
print(train_df.shape)
test_df = pd.read_csv(TEST_CSV)
print(test_df.shape)
for q in ['question1', 'question2']:
	train_df[q + '_n'] = train_df[q]
	test_df[q + '_n'] = test_df[q]

test_df = test_df[train_df.columns]

use_w2v = True
print('-------------')
# print(train_df.head())
# print(test_df.head())
train_size= train_df.shape[0]
print('train size: {}'.format(train_size))
print('-------------')
if BUILD_EMBED == True:
	full_df = train_df.append(test_df, ignore_index=True)
	full_df, embeddings = make_w2v_embeddings(full_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)
	print("sentences embedded")

else:
	# full_df= pd.read_csv('./data/full_embeddings_A1.csv')
	# embeddings = np.load('./data/embeddings/embedding_matrix_A1.npy')
	print('embeddings loaded')
train_df = full_df.iloc[:train_size].copy()
test_df = full_df.iloc[train_size:].copy()
print('--------------------------')
# print(train_df.head())
# print(test_df.head())
print('--------------------------')
# test_df, embeddingsx = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)
# print("sentences embedded")
# test_df.to_csv('./data/test_embeddings.csv', index= False)

# Split to train validation
validation_size = 0.2
# training_size = len(train_df) - validation_size

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state= 42)

X_test = test_df[['question1_n', 'question2_n']]
Y_test = test_df['is_duplicate']

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)
X_test= split_and_zero_padding(X_test, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values
# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# --

# Model variables
gpus = 1
batch_size = 1024 * gpus
n_epoch = 40
n_hidden = 50

# Define the shared model

with tf.device('/gpu:0'):
	set_session(tf.Session(config=config))
	x = Sequential()
	x.add(Embedding(len(embeddings), embedding_dim,
					weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
	# CNN
	# x.add(Conv1D(250, kernel_size=5, activation='relu'))
	# x.add(GlobalMaxPool1D())
	# x.add(Dense(250, activation='relu'))
	# x.add(Dropout(0.3))
	# x.add(Dense(50, activation='sigmoid'))
	# LSTM
	x.add(LSTM(n_hidden))

	shared_model = x

	# The visible layer
	left_input = Input(shape=(max_seq_length,), dtype='int32')
	right_input = Input(shape=(max_seq_length,), dtype='int32')

	# Pack it all up into a Manhattan Distance model
	malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
	model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
	model.summary()
	shared_model.summary()

############################

	# Start training
	training_start_time = time()
	malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
							   batch_size=batch_size, epochs=3,
							   validation_data=([X_validation['left'], X_validation['right']], Y_validation))
	training_end_time = time()

	print('INITIAL SCORES')
	print('--------------------------')
	preds = model.predict([X_validation['left'], X_validation['right']], batch_size=batch_size )
	predsx = preds.copy()
	predsx[predsx>0.4] = 1
	predsx[predsx<=0.4] = 0

	print("DEV F1 Score: {},\t DEV AUC ROC : {},\t DEV ACC: {}".format(f1_score(Y_validation, predsx), roc_auc_score(Y_validation, preds), accuracy_score(Y_validation, predsx) ))


	preds = model.predict([X_test['left'], X_test['right']], batch_size=batch_size )
	predsx = preds.copy()
	predsx[predsx>0.4] = 1
	predsx[predsx<=0.4] = 0

	print("TEST F1 Score: {},\t TEST AUC ROC : {},\t TEST ACC: {}".format(f1_score(Y_test, predsx), roc_auc_score(Y_test, preds), accuracy_score(Y_test, predsx) ))

######################################

	training_start_time = time()
	malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
							   batch_size=batch_size, epochs=n_epoch,
							   validation_data=([X_validation['left'], X_validation['right']], Y_validation))
	training_end_time = time()
	model.save('./data/SiameseLSTM_epoch25.h5')

	print('FINAL SCORES')
	print('--------------------------')

	preds = model.predict([X_validation['left'], X_validation['right']], batch_size=batch_size )
	thres = 0.4
	predsx = preds.copy()
	predsx[predsx>thres] = 1
	predsx[predsx<=thres] = 0
	dev_score= "DEV F1 Score: {},\t DEV AUC ROC : {},\t DEV ACC: {}".format(f1_score(Y_validation, predsx), roc_auc_score(Y_validation, preds), accuracy_score(Y_validation, predsx) )
	print(dev_score)

	preds = model.predict([X_test['left'], X_test['right']], batch_size=batch_size )
	predsx = preds.copy()
	predsx[predsx>thres] = 1
	predsx[predsx<=thres] = 0

	test_score= "TEST F1 Score: {},\t TEST AUC ROC : {},\t TEST ACC: {}".format(f1_score(Y_test, predsx), roc_auc_score(Y_test, preds), accuracy_score(Y_test, predsx) )
	print(test_score)
	acc_scores =[]
	thr = 0.4
	acc_scores.append(accuracy_score(Y_test, int_pred(preds, thr)))
	sv_scores = "ACC SCORES : {}".format(str(acc_scores))
	print(sv_scores)

	final_preds = np.array(list(zip(Y_test, preds, predsx)))
	np.save('./data/preds/predsLSTM_'+str(train_file)+ '.npy', final_preds)
	print('--------------------------')

	print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

	fo = open('./data/scores_lstm/scoresLSTM_'+str(train_file)+ '.txt', 'w')
	fo.write(train_file+'\n')
	fo.write('-----------\n')
	fo.write(sv_scores+'\n')
	fo.write('-----------\n')
	fo.write(test_score+'\n')
	fo.close()

print(str(malstm_trained.history['val_acc'][-1])[:6] + "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")
print(train_df.shape)
print(train_file)
