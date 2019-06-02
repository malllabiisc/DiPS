import argparse


def build_parser():
    # Data loading parameters
    parser = argparse.ArgumentParser(description='Run paraphrase single model')


    # Mode specifications
    parser.add_argument('-mode', type=str, required=True, choices=['train', 'decode'], help='Modes: train, decode')
    parser.add_argument('-debug', action='store_true', help='Operate on debug mode')
    parser.add_argument('-slam', type=float, default=0.5, help='Lambda Value for Submod')
    parser.add_argument('-a1', type=float, default=1.0, help='Lambda Value for Submod')
    parser.add_argument('-a2', type=float, default=1.0, help='Lambda Value for Submod')
    parser.add_argument('-b1', type=float, default=1.0, help='Lambda Value for Submod')
    parser.add_argument('-b2', type=float, default=1.0, help='Lambda Value for Submod')


    parser.add_argument('-selec', type=str, default='normal', choices=['normal', 'submod', 'random'], help='Subset Selection Method')
    parser.add_argument('-dataset', type=str, default='quora', choices=['quora', 'twitter'], help='Dataset to use')

    # Run name should just be alphabetical word (no special characters to be included)
    parser.add_argument('-run_name', type=str, default='DiPS', help='Enter the run name')
    parser.add_argument('-rev_seq', action='store_true', help='Send in the reversed sequence')
    parser.add_argument('-display_freq', type=int, default=200, help='number of batches after which to display loss')


    # Input files
    parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')
    parser.add_argument('-res_file', type=str, default='generations.txt', help='File name to save results in')
    parser.add_argument('-res_folder', type=str, default='Generations', help='Folder name to save results in')
    parser.add_argument('-out_dir', type=str, default='out', help='Out Dir')
    parser.add_argument('-len_sort', action="store_true", help='Sort based on length')


    # Device Configuration
    parser.add_argument('-gpu', type=str, required=True, help='Specify the gpu to use')
    parser.add_argument('-seed', type=int, default=1123, help='Default seed to set')
    parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

    # Dont modify ckpt_file
    # If you really want to then assign it a name like abc_0.pth.tar (You may only modify the abc part and don't fill in any special symbol. Only alphabets allowed
    parser.add_argument('-ckpt_file', type=str, default='s2s_0.pth.tar', help='Checkpoint file name')
    parser.add_argument('-date_fmt', type=str, default='%Y-%m-%d-%H:%M:%S', help='Format of the date')


    # Seq2Seq parameters
    parser.add_argument('-cell_type', type=str, default='lstm', help='RNN cell for encoder and decoder, default: lstm')
    parser.add_argument('-use_attn', action='store_true', help='To use attention mechanism?')
    parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
    parser.add_argument('-hidden_size', type=int, default=384, help='Number of hidden units in each layer')
    parser.add_argument('-depth', type=int, default=3, help='Number of layers in each encoder and decoder')
    parser.add_argument('-emb_size', type=int, default=256, help='Embedding dimensions of encoder and decoder inputs')
    parser.add_argument('-beam_width', type=int, default=10, help='Specify the beam width for decoder')
    parser.add_argument('-max_length', type=int, default=20, help='Specify max decode steps: Max length string to output')
    parser.add_argument('-s2sdprate', type=float, default=0.3, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
    parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
    parser.add_argument('-bidirectional', action='store_true', help='Initialization range for seq2seq model')
    parser.add_argument('-use_word2vec', action='store_true', help='Initialization Embedding matrix with word2vec vectors')
    parser.add_argument('-word2vec_bin', type=str, default='data/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')
    parser.add_argument('-train_word2vec', action='store_true', help='Binary file of word2vec')
    # parser.add_argument('-div_beam',            action='store_true',                                                            help='Perform diverse beam search')
    # parser.add_argument('-div_lam',             type=float,                     default=0.3,                                    help='lambda for diverse beam')
    # parser.add_argument('-div_gps',             type=int,                       default=5,                                      help='Groups in diverse beam')


    # Training parameters
    parser.add_argument('-lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-max_epochs', type=int, default=200, help='Maximum # of training epochs')
    parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
    parser.add_argument('-tfr', type=float, default=0.9, help='Teacher forcing ratio')


    return parser
