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
    parser = argparse.ArgumentParser(description='Run paraphrase seq2seq model')

    parser.add_argument('-train_file',       type=str,               default='data/yahoo/sub_train200.csv')
    parser.add_argument('-dataset',       type=str,               default='yahoo')
    parser.add_argument('-mode',       type=str,               default='generate')
    parser.add_argument('-data_dir',       type=str,               default='yahoos_submod')
    parser.add_argument('-aug_dir',       type=str,               default='aug_out')
    parser.add_argument('-beam_width',       type=int,               default=5)
    parser.add_argument('-gpu',       type=int,               default=0)
    parser.add_argument('-method',       type=str,               default='submod')


    args = parser.parse_args()

    beam = args.beam_width
    mode = str(args.mode)
    train_file = str(args.train_file)

    train_df = pd.read_csv(train_file)
    train_df['text'] = train_df['text'].apply(lambda x: x.lower())

    aug_dir = str(args.aug_dir)
    data_dir = str(args.data_dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        print('Data Dir Created')

    dataset = str(args.dataset)
    gpu = str(args.gpu)
    method = str(args.method)

    print('------------PARAMS------------')
    print('METHOD : {}'.format(method))
    print('DATA DIR : {}'.format(data_dir))
    print('DATASET : {}'.format(dataset))
    print('GPU : {}'.format(gpu))
    print('TRAIN FILE : {}'.format(train_file))
    print('------------------------------')

    labels = list(set(train_df['label'].values))
    if dataset == 'snips':
        labels =[0]

    print(train_df.head())
    print('LABELS : {}'.format(labels))
    temp_path = os.path.join(data_dir, 'samples')
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
        print('Temp Path Created')
    else:
        print('Data Dir Found')

    out_path = os.path.join(data_dir, 'out')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        print('Out Path Created')

    if mode == 'generate':
        # Create Directories to store source sentences for each label
        for i in labels:
            i = str(i)
            new_path = os.path.join(temp_path, i)
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
                print('Label Sample Path Created {}'.format(i))
            else:
                print('Label Path Found')

        # Read Train dataframe and save text files in the respective directories
        for i in labels:
            save_path = os.path.join(temp_path, str(i))
            train_label = train_df[train_df['label'] == i].copy()

            all_text = train_label[['text']]

            file_name = dataset+'_'+str(i)+ '.csv'
            all_text.to_csv(os.path.join(save_path, file_name), index= None , header=None)

        # Generate paraphrases for each label
        for i in labels:
            i = str(i)
            print('WORKING ON LABEL {}'.format(i))
            new_path = os.path.join(out_path, i)
            if not os.path.isdir(new_path):
                os.mkdir(new_path)

            load_file = dataset+'_'+str(i)+ '.csv'
            load_path = os.path.join(temp_path, i , load_file)
            beam_path = os.path.join(new_path, str(beam) )
            if not os.path.isdir(beam_path):
                os.mkdir(beam_path)
                print('Beam Path Created : {}'.format(beam))

            command = 'python -m src.run_model -mode decode -test_file_src {} -use_gpu {} -beam_width {} -selec {} -slam 0.5 -out_dir {}'.format(load_path, gpu, beam, method, beam_path)

            print('Running Paraphraser')
            os.system(command)


        print('Generated for all labels')

    else:
        train_df = train_df[['text', 'label']]
        aug_path = os.path.join(aug_dir, data_dir)
        if not os.path.isdir(aug_path):
            os.mkdir(aug_path)
            print('AUG Dir Created')

        data =[]
        beam =str(beam)

        for label in labels:
            label = str(label)
            out_dir = os.path.join(out_path, label, beam)
            gen_files = glob(out_dir + '/*')
            file = gen_files[0]
            res = np.load(file)
            req_data = return_data(res, label)
            data+= req_data

        aug_df = pd.DataFrame(data, columns= ['text', 'label'])
        new_df = train_df.append(aug_df)
        print(train_df.shape, new_df.shape)
        save_file = 'train_' + str(data_dir) + '_'+ str(beam)+'.csv'
        save_path =os.path.join(aug_path, save_file)
        new_df.to_csv(save_path, index = False)
        print('{} generations for beam {} saved at {}'.format(data_dir, beam, save_path))












