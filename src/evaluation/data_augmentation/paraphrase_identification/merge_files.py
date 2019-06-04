import pandas as pd
import numpy as np
import pickle
import pdb
import os
import argparse
from glob import glob

def return_data(res, label):
	resx = [x for ele in res for x in ele]
	data = list(zip(resx, [label]*len(resx) ))
	return data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run paraphrase seq 2 seq model')

	parser.add_argument('-train_file',       type=str,               default='data/quora/train_df.csv')
	parser.add_argument('-dataset',       type=str,               default='quora')
	parser.add_argument('-data_dir',       type=str,               default='./data/gen_files/')
	parser.add_argument('-aug_dir',       type=str,               default='data/merged_files')
	parser.add_argument('-src_gen',       type=str,               default='./data/quora/src_gen.txt')
	parser.add_argument('-beam_width',       type=int,               default=5)
	parser.add_argument('-method',       type=str,               default='submod')

	args = parser.parse_args()
	src = []
	src_gen= args.src_gen
	fo = open(src_gen, 'r')
	bw = args.beam_width
	method = args.method
	while True:
		line = fo.readline()
		if line:
			src.append(line.strip())
		else:
			break

	fo.close()
	print('Source Sentences: {}'.format(len(src)))
	print(src[:5])

	train_file = str(args.train_file)
	data_dir = str(args.data_dir)

	gen_files = glob(data_dir + '*'+method+'*')     # numpy files with generated paraphrases to be merged
	train_df = pd.read_csv(train_file)
	aug_dir = str(args.aug_dir)

	dataset = str(args.dataset)

	for file in gen_files:
		fname = '_'.join(file.split('/')[-1].split('_')[1:]).replace('.npy', '')

		print('------------PARAMS------------')
		print('DATA DIR : {}'.format(data_dir))
		print('TRAIN FILE : {}'.format(fname))
		print('------------------------------')

		res = np.load(file)
		try:
			all_pairs = [[src[i], res[i][j],1] for i in range(len(src)) for j in range(bw)]
		except:
			continue
		print('Number of Aug Samples {}'.format(len(all_pairs)))
		new_df = pd.DataFrame(all_pairs ,columns =['question1','question2', 'is_duplicate'])
		aug_df = train_df.append(new_df)
		aug_df = aug_df.sample(frac=1.0)
		aug_df.index = list(range(len(aug_df)))
		sv_path= os.path.join(aug_dir,'train_'+ fname+'_'+ str(bw) + '.csv')
		aug_df.to_csv(sv_path, index=None)
		print('File Saved at {}'.format(sv_path))













