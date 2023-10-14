import os, math, logging, pdb, sys
import time
from datetime import datetime
from attrdict import AttrDict

import torch, random
import numpy as np
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

try:
    import cPickle as pickle
except ImportError:
    import pickle

from src.model import s2s
from src.helper import *
from src.dataloader import *
from src.args import build_parser

from collections import OrderedDict
# python -m src.main -mode decode -gpu 1 -beam_width 20 -selec normal



def read_files(args, logger):
    if args.mode == 'train':
        logger.info('Training and Validation data loading..')
        train_set           = ParaphraseDataset(args.dataset, 'train', args.max_length, args.debug, True)
        val_set             = ParaphraseDataset(args.dataset, 'val', args.max_length, args.debug)

        if args.len_sort:
            train_dataloader    = DataLoader(train_set, batch_size = args.batch_size, shuffle=False, num_workers=5)
            val_dataloader      = DataLoader(val_set, batch_size = args.batch_size, shuffle=False, num_workers=5)
        else:
            train_dataloader    = DataLoader(train_set, batch_size = args.batch_size, shuffle=True, num_workers=5)
            val_dataloader      = DataLoader(val_set, batch_size = args.batch_size, shuffle=True, num_workers=5)

        logger.info('Training and Validation data loaded!')

        return train_dataloader, val_dataloader

    elif args.mode == 'decode':
        logger.info('Test data loading..')
        test_set            = ParaphraseDataset(args.dataset, 'test', args.max_length, args.debug)
        test_dataloader     = DataLoader(test_set, batch_size = args.batch_size, shuffle=False, num_workers=5)
        logger.info('Test data loaded!')

        return test_dataloader

    else:
        raise Exception('{} is not a valid mode'.format(args.mode))



# Implement training procedure + Validation step
def train(model, train_dataloader, val_dataloader, voc, device, args, logger, ep_offset = 0, min_val_loss=1e8, max_val_bleu=0.0):
    logger.info('Training Started!!')
    writer = SummaryWriter(os.path.join('Logs', args.run_name))

    for ep in range(args.max_epochs):

        od = OrderedDict()
        od['Epoch'] = ep + ep_offset
        print_log(logger, od)

        batch_num = 1
        train_loss_epoch = 0
        val_loss_epoch = 0

        # Start train mode (Update weights)
        model.train()
        # Start batch-wise training procedure
        for pairs in train_dataloader:
            if batch_num % args.display_freq == 0:

                od = OrderedDict()
                od['Batch'] = batch_num
                od['Loss'] = loss
                print_log(logger, od)

            src_tens  = indicesFromSentences(voc, pairs['src'], args.max_length)
            tgt_tens  = indicesFromSentences(voc, pairs['tgt'], args.max_length)

            src_tens, src_len, tgt_tens, tgt_len, _ = process_batch(src_tens, tgt_tens, voc, device)

            loss = model.trainer(src_tens, src_len, tgt_tens, tgt_len)

            train_loss_epoch += loss
            batch_num += 1

        writer.add_scalar('loss/train_loss', train_loss_epoch, ep + ep_offset)

        # Run Validation experiment
        bleu_score_epoch, val_loss_epoch = run_validation(args, model, val_dataloader, voc, device, ep + ep_offset, logger)

        if bleu_score_epoch[0] > max_val_bleu:
            min_val_loss = val_loss_epoch
            max_val_bleu = bleu_score_epoch[0]
            state = {
                'epoch' : ep + ep_offset,
                'model_state_dict': model.state_dict(),
                'voc': model.voc,
                'optimizer_state_dict': model.optimizer.state_dict(),
                'train_loss' : train_loss_epoch,
                'val_loss' : min_val_loss,
                'bleu' : max_val_bleu
            }
            save_checkpoint(state, ep+ep_offset, logger, os.path.join('Model', args.run_name, args.ckpt_file), args)

        writer.add_scalar('loss/val_loss', val_loss_epoch, ep + ep_offset)
        # Validation code after each epoch
        od = OrderedDict()
        od['Epoch'] = ep + ep_offset
        od['Train_loss'] = train_loss_epoch
        od['Val_loss'] = val_loss_epoch
        print_log(logger, od)

    writer.export_scalars_to_json(os.path.join('Logs', args.run_name, 'all_scalars.json'))
    writer.close()
    logger.info('Training Completed!')



def run_validation(args, model, val_dataloader, voc, device, ep, logger):
    batch_num = 1
    refs = []
    hyps = []
    val_loss_epoch = 0

    # Switch to evaluation mode for validation
    model.eval()

    logger.info('Sample Generations')
    logger.info('==========================================')
    logger.info('==========================================')
    logger.info('==========================================')

    for pairs in val_dataloader:
        src_tens  = indicesFromSentences(voc, pairs['src'], args.max_length)
        tgt_tens  = indicesFromSentences(voc, pairs['tgt'], args.max_length)

        src_tens, src_len, tgt_tens, tgt_len, _   = process_batch(src_tens, tgt_tens, voc, device)
        val_loss, decoder_output, decoder_attn = model.greedy_decode(src_tens, src_len, tgt_tens, tgt_len, True)

        src_sents       = indicesToSentences(voc, src_tens, True)
        tgt_sents       = indicesToSentences(voc, tgt_tens, True)
        val_loss_epoch += val_loss

        refs += [[' '.join(tgt_sents[i])] for i in range(tgt_tens.size(1))]
        hyps += [' '.join(decoder_output[i]) for i in range(src_tens.size(1))]
        # Save model

        if batch_num % args.display_freq == 0:
            for i in range(10):
                try:
                    od = OrderedDict()
                    logger.info('-----------------------')
                    od['Source'] = ' '.join(src_sents[i])
                    print_log(logger, od)

                    od = OrderedDict()
                    od['Target'] = ' '.join(tgt_sents[i])
                    print_log(logger, od)

                    od = OrderedDict()
                    od['Paraphrase'] = ' '.join(decoder_output[i])
                    print_log(logger, od)
                    logger.info('-----------------------')
                except:
                    break

        batch_num += 1

    logger.info('==========================================')
    logger.info('==========================================')
    logger.info('==========================================')

    bleu_score_epoch = bleu_scorer(refs, hyps)
    od = OrderedDict()
    od['Epoch'] = ep
    od['BLEU'] = bleu_score_epoch
    print_log(logger, od)

    return bleu_score_epoch, val_loss_epoch



# Greedy Decoding
def decode_greedy(model, test_dataloader,  voc, device, args, logger):
    logger.info('Test Generations')
    result_dir = os.path.join(args.res_folder, args.run_name)
    results_file = os.path.join(result_dir, '{}_{}'.format(args.res_file, args.beam_width))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(results_file, 'w') as f:
        for pairs in test_dataloader:
            src_tens  = indicesFromSentences(voc, pairs['src'], args.max_length)
            tgt_tens  = indicesFromSentences(voc, pairs['tgt'], args.max_length)

            src_tens, src_len, tgt_tens, tgt_len, orig_order = process_batch(src_tens, tgt_tens, voc, device)

            decoder_output = model.greedy_decode(src_tens, src_len)
            # Get back original order
            decoder_output = [decoder_output[i] for i in orig_order]

            src_sents = indicesToSentences(voc, src_tens, True)
            src_sents = [src_sents[i] for i in orig_order]

            for i in range(len(src_sents)):
                f.write('Sentence: {} \n'.format(' '.join(src_sents[i])))
                f.write('Beam 0 : {}\n'.format(' '.join(decoder_output[i])))
                f.write('-----------------\n')

                logger.info('Sentence: {} '.format(' '.join(src_sents[i])))
                logger.info('Beam 0 : {}'.format(' '.join(decoder_output[i])))
                logger.info('-------------------------')

    logger.info('Decoding Complete!!')



# Implement decoding procedure
def decode_beam(model, test_dataloader,  voc, device, args, logger, smethod, data_sub):
    logger.info('Test Generations')
    result_dir = os.path.join(args.res_folder, args.run_name)
    results_file = os.path.join(result_dir, '{}_{}'.format(args.res_file, args.beam_width))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    all_results = []
    all_src =[]
    batch_num = 1


    st_time = time.time()
    with open(results_file, 'w') as f:
        for pairs in test_dataloader:
            logger.info('Processing batch : {}'.format(batch_num))
            src_tens  = indicesFromSentences(voc, pairs['src'], args.max_length)
            tgt_tens  = indicesFromSentences(voc, pairs['tgt'], args.max_length)

            src_tens, src_len, tgt_tens, tgt_len, orig_order = process_batch(src_tens, tgt_tens, voc, device)

            src_sents = indicesToSentences(voc, src_tens, True)

            if smethod =='normal':
                decoder_output = model.beam_decode(src_sents, src_tens, src_len, args.beam_width)
            else:
                decoder_output = model.beam_decode_sub(src_sents, src_tens, src_len, args.beam_width,
                                                       method= smethod, slam= args.slam, sparam = args.sparam)

            decoder_output = [decoder_output[i] for i in orig_order]
            src_sents = [src_sents[i] for i in orig_order]

            all_results+= decoder_output

            for i in range(len(src_sents)):
                all_src += [' '.join(src_sents[i])]

            for i in range(len(src_sents)):
                logger.info('Sentence: {} '.format(' '.join(src_sents[i])))
                for j in range(len(decoder_output[i]) ):
                    logger.info('Beam {} : {}'.format(j+1, decoder_output[i][j]))
                logger.info('-------------------------')

                f.write('Sentence: {} \n'.format(' '.join(src_sents[i])))
                for j in range(len(decoder_output[i]) ):
                    f.write('Beam {} : {}'.format(j+1, decoder_output[i][j]))
                f.write('-----------------\n')

            batch_num += 1

    etime = (time.time() - st_time)/60.0
    print('Time Taken for decoding: {}'.format(etime))

    all_results= np.array(all_results)
    all_src = np.array(all_src)
    final_res = []
    for sk in range(len(all_src)):
        td=[]
        for gen in all_results[sk]:
            td.append([all_src[sk], gen])
        final_res.append(td)
    final_res = np.array(final_res)



    param_str = [str(s) for s in args.sparam]
    param_str= '_'.join(param_str)

    outdir = str(args.out_dir)

    if smethod == 'submod':
        res_save_path= os.path.join(outdir, 'results_{}_{}_{}_{}'.format(smethod, data_sub, args.slam, param_str))
        fres_save_path= os.path.join(outdir, 'fres_{}_{}_{}_{}'.format(smethod, data_sub, args.slam, param_str))
    else:
        res_save_path= os.path.join(outdir, 'results_{}_{}'.format(smethod, data_sub))
        fres_save_path= os.path.join(outdir, 'fres_{}_{}'.format(smethod, data_sub))

    np.save(res_save_path, all_results)
    np.save(fres_save_path, final_res)
    print(all_results.shape)
    print(final_res.shape)
    print('Output Saved at {}'.format(res_save_path))
    print('Output Saved at {}'.format(fres_save_path))


def create_vocab_dict(args, voc, train_dataloader):
    for pairs in train_dataloader:
        for src, tgt in zip(pairs['src'], pairs['tgt']):
            voc.addSentence(src)
            voc.addSentence(tgt)
    voc.most_frequent(args.vocab_size)
    assert len(voc.w2id) == voc.nwords
    assert len(voc.id2w) == voc.nwords
    return voc


def main():

    # Parse arguments
    parser = build_parser()
    args = parser.parse_args()
    args.mode = args.mode.lower()

    # div_gps = args.div_gps
    # div_lam = args.div_lam
    if args.mode == 'train':
        if len(args.run_name.split()) == 0:
            args.run_name = datetime.fromtimestamp(time.time()).strftime(args.date_fmt)
    else:
        args.run_name = args.run_name

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    smethod = str(args.selec)
    data_sub = str(os.path.join('data', args.dataset, 'test', 'src.txt')).split('/')[-1].split('.')[0]
    slam = args.slam

    a1 = args.a1
    a2 = args.a2
    b1 = args.b1
    b2 = args.b2
    sparam = [a1,a2,b1,b2]

    outdir= str(args.out_dir)

    # GPU initialization
    device = gpu_init_pytorch(args.gpu)

    log_folder_name = os.path.join('Logs', args.run_name)
    create_save_directories('Logs', 'Model', args.run_name)
    logger = get_logger(__name__, args.run_name, args.log_fmt, logging.INFO, os.path.join(log_folder_name, 's2s.log'))


    if args.mode == 'train':
        train_dataloader, val_dataloader = read_files(args, logger)
        logger.info('Creating vocab ...')

        voc = Voc(args.dataset)
        voc = create_vocab_dict(args, voc, train_dataloader)

        logger.info('Vocab created with number of words = {}'.format(voc.nwords))
        logger.info('Saving Vocabulary file')

        with open(os.path.join('Model', args.run_name, 'vocab.p'), 'wb') as f:
            pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info('Vocabulary file saved in {}'.format(os.path.join('Model', args.run_name, 'vocab.p')))
    else:
        test_dataloader = read_files(args, logger)
        logger.info('Loading Vocabulary file')

        with open(os.path.join('Model', args.run_name, 'vocab.p'), 'rb') as f:
            voc = pickle.load(f)

        logger.info('Vocabulary file Loaded from {}'.format(os.path.join('Model', args.run_name, 'vocab.p')))



    # Get Checkpoint, return None if no checkpoint present
    checkpoint = get_latest_checkpoint('Model', args.run_name, logger)



    if args.mode == 'train':
        if checkpoint == None:
            logger.info('Starting a fresh training procedure')
            ep_offset = 0
            min_val_loss = 1e8
            max_val_bleu = 0.0
            config_file_name = os.path.join('Model', args.run_name, 'config.p')

            if args.use_word2vec:
                args.emb_size = 300

            model = s2s(args, voc, device, logger)

            with open(config_file_name, 'wb') as f:
                pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            config_file_name = os.path.join('Model', args.run_name, 'config.p')

            with open(config_file_name, 'rb') as f:
                args = AttrDict(pickle.load(f))

            if args.use_word2vec:
                args.emb_size = 300

            model = s2s(args, voc, device, logger)

            ep_offset, train_loss, min_val_loss, max_val_bleu, voc = load_checkpoint(model, args.mode, checkpoint, logger, device)

            logger.info('Resuming Training From ')
            od = OrderedDict()
            od['Epoch'] = ep_offset
            od['Train_loss'] = train_loss
            od['Validation_loss'] = min_val_loss
            od['Validation_Bleu'] = max_val_bleu
            print_log(logger, od)
            ep_offset += 1

        # Call Training function
        train(model, train_dataloader, val_dataloader, voc, device, args, logger, ep_offset, min_val_loss, max_val_bleu)
    else:
        if checkpoint == None:
            logger.info('Cannot decode because of absence of checkpoints')
            sys.exit()
        else:
            config_file_name = os.path.join('Model', args.run_name, 'config.p')
            beam_width = args.beam_width
            gpu = args.gpu

            with open(config_file_name, 'rb') as f:
                args = AttrDict(pickle.load(f))
                args.beam_width = beam_width
                args.gpu = gpu
                # args.div_beam = div_beam
                # args.div_gps = div_gps
                # args.div_lam = div_lam

            if args.use_word2vec:
                args.emb_size = 300

            args.slam = slam
            args.sparam = sparam
            args.out_dir = outdir

            model = s2s(args, voc, device, logger)

            ep_offset, train_loss, min_val_loss, max_val_bleu, voc = load_checkpoint(model, args.mode, checkpoint, logger, device)

            logger.info('Decoding from')
            od = OrderedDict()
            od['Epoch'] = ep_offset
            od['Train_Loss'] = train_loss
            od['Validation_Loss'] = min_val_loss
            od['Validation_Bleu'] = max_val_bleu
            print_log(logger, od)

        if args.beam_width == 1:
            decode_greedy(model, test_dataloader, voc, device, args, logger)
        else:
            decode_beam(model, test_dataloader, voc, device, args, logger, smethod, data_sub)


if __name__ == '__main__':
    main()
