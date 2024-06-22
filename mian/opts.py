import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='Our model options')

    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'choose device to use cpu or gpu')

    parser.add_argument('--arch', default = 'vit_tiny_patch16_224_add_gcn', type = str, 
                        help = 'Name of model to train')

    parser.add_argument('--num_classes', action = 'store', type = int, default = 5,
                        help = 'number of classifications')
    
    # params for correlation matrix
    parser.add_argument('--prob_threshold', action = 'store', type = float, default = 0.3,
                        help = 'probability thresholds (tau) are used to binarise the correlation matrix')
    
    parser.add_argument('--scaling_factor', action = 'store', type = float, default = 0.25,
                        help = 'scaling_factor (eta) reduces the matrix to a smaller range')
    # path
    parser.add_argument('--adj_file_path', action = 'store', type = str, default = './utils/cavitation_train_adj/cavitation_train_split_466944_10_adj.pkl',
                        help = 'path of correlation matrix by data-driven way')
    
    parser.add_argument('--word_embedding', action = 'store', type = str, default = './utils/cavitation_glove_word2vec.pkl',
                        help = 'path of correlation matrix by data-driven way')

    parser.add_argument('--train_root', action = 'store', type = str, default = '/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Train/STFT/466944_10',   
                        help = 'root path of training data')

    parser.add_argument('--train_label_path', action = 'store', type = str, default = '/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Label/multi_label/train_split_466944_10.csv',
                        help = 'path of training data label')
    
    parser.add_argument('--test_root', action = 'store', type = str, default = '/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Test/STFT/466944_10',
                        help = 'root path of test data')
                        
    parser.add_argument('--test_label_path', action = 'store', type = str, default = '/home/deepthinkers/samson/yusha_workspace/Cavitation2017_STFT/Label/multi_label/test_split_466944_10.csv',
                        help = 'path of test data label')
    
    # params for network
    parser.add_argument('--batch_size', action = 'store', type = int, default = 64,
                        help = 'mimi-batch size (default: 16)')
    
    parser.add_argument('--learning_rate', action = 'store', type = float, default = 0.1,
                        help = 'learning rate of the network')
    
    parser.add_argument('--learning_rate_pretrained', action = 'store', type = float, default = 0.1,
                        help = 'learning rate for pre-trained layers')
    
    parser.add_argument('--momentum', action = 'store', type = float, default = 0.9,
                        help = 'momentum')

    parser.add_argument('--weight_decay', action = 'store', type = float, default = 1e-4,
                        help = 'weight decay (default: 1e-4)')

    parser.add_argument('--epoches', action = 'store', type = int, default = 100,
                        help = 'number of total epoches to run')
    
    parser.add_argument('--expectedacc', action = 'store', type = float, default = 0.9,
                        help = 'expected accuracy of the test set for model saving')

    parser.add_argument('--n_threads', action = 'store', type = int, default = 1,
                        help = 'number of threads for multi-thread loading')

    parser.add_argument('--train_batch_shuffle', action = 'store', type = bool, default = True,
                        help = 'shuffle input batch for training data')

    parser.add_argument('--test_batch_shuffle', action = 'store', type = bool, default = False,
                        help = 'shuffle input batch for training data')

    parser.add_argument('--train_drop_last', action = 'store', type = bool, default = False,
                        help = 'drop the remaining of the batch if the size does not match minimum batch size')

    parser.add_argument('--test_drop_last', action = 'store', type = bool, default = False,
                        help = 'drop the remaining of the batch if the size does not match minimum batch size')
    
    args = parser.parse_args()
    argsDict = args.__dict__
    # save pars  
    with open("pars_seeting.json",'w') as f:
        f.writelines('-----------------------start---------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('-----------------------end-----------------------' + '\n')
    return args
