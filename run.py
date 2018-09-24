'''
Created on Dec 9, 2015

@author: donghyun
'''
import argparse
import sys
import os
import glob
import itertools
import numpy as np
import cPickle as pickl

from models import ConvCAEMF
from models import ConvMF
from models import CAEMF
from models import MF
from models import stacking_CNN_CAE
from models import NN_stacking_CNN_CAE
from models import Raw_att_CNN_concat
from data_manager import Data_Factory
from rec_eval.lib.evaluator import Evaluator
from util import Logger, print_helper



parser = argparse.ArgumentParser()

# Option for pre-processing data
parser.add_argument("-c", "--do_preprocess", action='store_true',
                    help="True or False to preprocess raw data for ConvMF (default = False)", default=False)
parser.add_argument("-r", "--raw_rating_data_path", type=str,
                    help="Path to raw rating data. data format - user id::item id::rating")
parser.add_argument("-i", "--raw_item_document_data_path", type=str,
                    help="Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...")
parser.add_argument("-m", "--min_rating", type=int,
                    help="Users who have less than \"min_rating\" ratings will be removed (default = 1)", default=1)
parser.add_argument("-l", "--max_length_document", type=int,
                    help="Maximum length of document of each item (default = 300)", default=300)
parser.add_argument("--max_df", type=float,
                    help="Threshold to ignore terms that have a document frequency higher than the given value (default = 1.0)",
                    default=1.0)
parser.add_argument("-s", "--vocab_size", type=int,
                    help="Size of vocabulary (default = 8000)", default=8000)
parser.add_argument("-t", "--split_ratio", type=float,
                    help="Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively (default = 0.2)",
                    default=0.2)

# Option for pre-processing data and running ConvMF
parser.add_argument("-d", "--data_path", type=str,
                    help="Path to training, valid and test data sets")
parser.add_argument("--splits_dir", type=str,
                    help="Path to the generated folds directory")
parser.add_argument("-a", "--aux_path", type=str, help="Path to R, D_all sets")
parser.add_argument("--fold_num", "-f", type=int, help="The number of folds to be generated. Default is 5",
                    choices=[5, 10], required=True)

# Option for running ConvMF
parser.add_argument("-o", "--res_dir", type=str,
                    help="Path to ConvMF's result")
parser.add_argument("-e", "--emb_dim", type=int,
                    help="Size of latent dimension for word vectors (default: 200)", default=200)
parser.add_argument("-p", "--pretrain_w2v", type=str,
                    help="Path to pretrain word embedding model  to initialize word vectors")
parser.add_argument("-g", "--give_item_weight",
                    help="Use item-specific weight, check Donghyun Kim '17 paper (default = False)",
                    action='store_true')
parser.add_argument("-k", "--dimension", type=int,
                    help="Dimension of users and items latent vector(default: 200)", default=200)
parser.add_argument("-u", "--lambda_u", type=float,
                    help="Value of user regularizer")
parser.add_argument("-v", "--lambda_v", type=float,
                    help="Value of item regularizer")
parser.add_argument("-n", "--max_iter", type=int,
                    help="Value of max iteration (default: 200)", default=200)
parser.add_argument("-w", "--num_kernel_per_ws", type=int,
                    help="Number of kernels per window size for CNN module (default: 100)", default=100)
parser.add_argument("--content_mode", type=str,
                    choices=['cnn', 'cnn_cae', 'cae', 'mf', 'stacking', 'nn_stacking','maria','raw_att_cnn'],
                    help="Content to be used, CNN: textual content, CAE: auxiliary item features", default='cnn')
parser.add_argument("--join_mode", type=str, choices=['concat', 'transfer'],
                    help="Approach used to joing the outputs of CNN and CAE (default: concat)", default='concat')
parser.add_argument("--att_dim", type=int,
                    help="Dimension of attributes latent vector (default: 50)", default=50)
parser.add_argument("--grid_search",
                    help="Run grid search to tune the hyperparameters (default = False)",action='store_true')
parser.add_argument("-lr", "--learning_rate", type=float,
                    help="learning rate used for ensemble")
parser.add_argument("--use_CAE",
                    help="Use CAE in the CNN||CAE model instead of CNN||FC (default = False)",action='store_true')

args = parser.parse_args()
grid_search = args.grid_search
do_preprocess = args.do_preprocess
data_path = args.data_path
aux_path = args.aux_path
splits_dir = args.splits_dir
fold_num = args.fold_num
if data_path is None:
    sys.exit("Argument missing - data_path is required")
if aux_path is None:
    sys.exit("Argument missing - aux_path is required")

data_factory = Data_Factory()

if do_preprocess:
    path_rating = args.raw_rating_data_path
    path_itemtext = args.raw_item_document_data_path
    min_rating = args.min_rating
    max_length = args.max_length_document
    max_df = args.max_df
    vocab_size = args.vocab_size
    split_ratio = args.split_ratio

    print "=================================Preprocess Option Setting================================="
    print "\trating data path - %s" % path_rating
    print "\tdocument data path - %s" % path_itemtext
    print "\tmin_rating: %d\n\tmax_length_document: %d\n\tmax_df: %.1f\n\tvocab_size: %d" \
          % (min_rating, max_length, max_df, vocab_size)
    print "\t{}".format(
        "split_ratio: %.1f" % split_ratio if splits_dir is None else "Splits: using pre-generated folds")
    print "\tsaving preprocessed aux  to - %s" % aux_path
    print "\tsaving preprocessed splits to - %s" % data_path
    print "==========================================================================================="

    R, D_all = data_factory.preprocess(
        path_rating, path_itemtext, min_rating, max_length, max_df, vocab_size)
    data_factory.save(aux_path, R, D_all)
    # Read training, test, and valid sets from the generated folds
    # data_factory.generate_train_valid_test_file_from_R(
    #     data_path, R, split_ratio)
    for f in range(1, fold_num + 1):
        fold_res_dir = os.path.join(data_path, 'fold-{}'.format(f))
        if not os.path.exists(fold_res_dir):
            os.makedirs(fold_res_dir)
        print "==========================================================================================="
        print "Running on fold %d" % (f)
        data_factory.generate_train_valid_test_from_ctr_split(os.path.join(splits_dir, 'fold-{}'.format(f)),
                                                              fold_res_dir)

elif not grid_search:

    # general params
    res_dir = args.res_dir
    dimension = args.dimension
    lambda_u = args.lambda_u
    lambda_v = args.lambda_v
    max_iter = args.max_iter
    give_item_weight = args.give_item_weight
    content_mode = args.content_mode
    join_mode = args.join_mode
    if join_mode == 'transfer':
        use_transfer_block = True
    else:
        use_transfer_block = False
    if lambda_u is None:
        sys.exit("Argument missing - lambda_u is required")
    if lambda_v is None:
        sys.exit("Argument missing - lambda_v is required")

    # R, D_all = data_factory.load(aux_path)
    R = data_factory.load_ratings(aux_path)
    # CNN params
    if 'cnn' in content_mode:
        emb_dim = args.emb_dim
        pretrain_w2v = args.pretrain_w2v
        max_length = args.max_length_document
        num_kernel_per_ws = args.num_kernel_per_ws

        D_all = data_factory.load_documents(aux_path)
        CNN_X = D_all['X_sequence']
        vocab_size = len(D_all['X_vocab']) + 1

        if pretrain_w2v is None:
            init_W = None
        else:
            init_W = data_factory.read_pretrained_word2vec(
                pretrain_w2v, D_all['X_vocab'], emb_dim)

    # CAE params
    if 'cae' in content_mode or content_mode in ['maria','raw_att_cnn']:
        att_dim = args.att_dim
        # Read item's attributes
        labels, features_matrix = data_factory.read_attributes(os.path.join(aux_path + 'paper_attributes.tsv'))
    # ensemble params
    if content_mode in ['stacking','nn_stacking','maria','raw_att_cnn']:
        if args.learning_rate is None:
            sys.exit("Argument missing - learning rate is required")
        lr = args.learning_rate

    if res_dir is None:
        sys.exit("Argument missing - res_dir is required")
    else:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    print "=================================== Option Setting==================================="
    print "\taux path - %s" % aux_path
    print "\tdata path - %s" % data_path
    print "\tresult path - %s" % res_dir
    print "\tdimension: %d\n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tmax_iter: %d\n" % (
        dimension, lambda_u, lambda_v, max_iter)
    print "\tContent: %s" % ( print_helper(content_mode))

    if 'cnn' in content_mode:
        print "\tnum_kernel_per_ws: %d\n\tpretrained w2v data path - %s" % (num_kernel_per_ws, pretrain_w2v)
    print('\tItem weight %s ' % ('Constant (a=1,b=0,01)' if not give_item_weight
                                 else 'Constant (a=1,b=0,01). And f(n)'))
    if 'cnn_cae' in content_mode:
        print '\tJoin CNN and CAE outputs method: %s' % ('Transfer block' if use_transfer_block else 'Concatenation')
    print "==========================================================================================="

    for f in range(1, fold_num + 1):
        train_user = data_factory.read_rating(
            os.path.join(data_path, 'fold-{}'.format(f), 'train-fold_{}-users.dat'.format(f)))
        train_item = data_factory.read_rating(
            os.path.join(data_path, 'fold-{}'.format(f), 'train-fold_{}-items.dat'.format(f)))
        # in case of training only on training and test sets
        if os.path.exists(os.path.join(data_path, 'fold-{}'.format(f), 'validation-fold_{}-users.dat'.format(f))):
            valid_user = data_factory.read_rating(
                os.path.join(data_path, 'fold-{}'.format(f), 'validation-fold_{}-users.dat'.format(f)))
        else:
            valid_user = None
        test_user = data_factory.read_rating(
            os.path.join(data_path, 'fold-{}'.format(f), 'test-fold_{}-users.dat'.format(f)))

        fold_res_dir = os.path.join(res_dir, 'fold-{}'.format(f))
        if not os.path.exists(fold_res_dir):
            os.makedirs(fold_res_dir)
        # train_user = data_factory.read_rating(glob.glob(data_path + '/train-fold_*-users.dat')[0])
        # train_item = data_factory.read_rating(glob.glob(data_path + '/train-fold_*-items.dat')[0])
        # valid_user = data_factory.read_rating(glob.glob(data_path + '/validation-fold_*-users.dat')[0])
        # test_user = data_factory.read_rating(glob.glob(data_path + '/test-fold_*-users.dat')[0])

        print "==========================================================================================="
        print ('Training on fold %d' % f)
        if content_mode == 'cnn_cae':

            ConvCAEMF(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
                      lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, vocab_size=vocab_size, init_W=init_W,
                      give_item_weight=give_item_weight, CNN_X=CNN_X, emb_dim=emb_dim,
                      num_kernel_per_ws=num_kernel_per_ws,max_len=max_length,
                      train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,
                      attributes_X=features_matrix, cae_output_dim=att_dim, use_transfer_block=use_transfer_block)
        elif content_mode == 'cnn':
            ConvMF(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
                   lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, vocab_size=vocab_size, init_W=init_W,max_len=max_length,
                   give_item_weight=give_item_weight, CNN_X=CNN_X, emb_dim=emb_dim, num_kernel_per_ws=num_kernel_per_ws,
                   train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)
        elif content_mode == 'cae':
            # attributes dimension must be euall to u, and v vectors dimension
            CAEMF(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
                  lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, cae_output_dim=dimension,
                  give_item_weight=give_item_weight,
                  train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,
                  attributes_X=features_matrix)
        elif content_mode == 'mf':
            MF(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
               lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension,
               give_item_weight=give_item_weight,
               train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)
        elif content_mode == 'stacking':
            CNN_theta = np.loadtxt(os.path.join(data_path, 'fold-{}'.format(f), 'CNN_theta.dat'.format(f)))
            CAE_gamma = np.loadtxt(os.path.join(data_path, 'fold-{}'.format(f), 'CAE_gamma.dat'.format(f)))
            stacking_CNN_CAE(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
                             lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension,
                             give_item_weight=give_item_weight, lr=lr, CNN_theta=CNN_theta, CAE_gamma=CAE_gamma,
                             train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user,
                             R=R)
        elif content_mode == 'nn_stacking':
            CNN_theta = np.loadtxt(os.path.join(data_path, 'fold-{}'.format(f), 'CNN_theta.dat'.format(f)))
            CAE_gamma = np.loadtxt(os.path.join(data_path, 'fold-{}'.format(f), 'CAE_gamma.dat'.format(f)))
            NN_stacking_CNN_CAE(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
                             lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension,
                             give_item_weight=give_item_weight, lr=lr, CNN_theta=CNN_theta, CAE_gamma=CAE_gamma,
                             train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user,
                             R=R)
        elif content_mode == 'maria':
            CNN_theta = np.loadtxt(os.path.join(data_path, 'fold-{}'.format(f), 'CNN_theta.dat'.format(f)))
            CAE_gamma = features_matrix
            NN_stacking_CNN_CAE(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
                                lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension,
                                give_item_weight=give_item_weight, lr=lr, CNN_theta=CNN_theta, CAE_gamma=CAE_gamma,
                                train_user=train_user, train_item=train_item, valid_user=valid_user,
                                test_user=test_user,
                                R=R)
        elif content_mode == 'raw_att_cnn':
            use_CAE = args.use_CAE
            Raw_att_CNN_concat(max_iter=max_iter, res_dir=fold_res_dir, state_log_dir=fold_res_dir,
                      lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, vocab_size=vocab_size, init_W=init_W,
                      give_item_weight=give_item_weight, CNN_X=CNN_X, emb_dim=emb_dim,
                      num_kernel_per_ws=num_kernel_per_ws,max_len=max_length,
                      train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,
                      attributes_X=features_matrix, use_CAE=use_CAE)

if grid_search:

    # To avoid long training time, it runs on 1 fold only.

    # general params

    res_dir = args.res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    sys.stdout = Logger(os.path.join(res_dir, 'log.txt'))
    dimension = args.dimension
    max_iter = args.max_iter
    give_item_weight = args.give_item_weight

    # Hyperparameters options
    lambda_u_list = [0.01, 0.1, 1]  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    lambda_v_list = [100, 1000]  # [0.01, 0.1, 1, 10, 100, 1000, 1000, 100000]
    confidence_mods = ['c', 'w']  # TODO: , 'user-dependant'] # c: constant, ud: user_dependent
    content_mods = ['cnn_cae']  # ['mf','cnn_cae','cnn','cae']  # ['cnn_cae','cnn']
    att_dims = [10, 20, 50, 100, 200]  # [10, 20, 50, 100, 200]
    join_mode = args.join_mode
    if join_mode == 'transfer':
        use_transfer_block = True
    else:
        use_transfer_block = False

    num_config = len(list(itertools.product(lambda_u_list, lambda_v_list, confidence_mods, content_mods)))
    if 'cae' or 'cnn_cae' in confidence_mods:
        num_config = (num_config * (len(att_dims) + 1)) / 2

    # CNN params
    emb_dim = args.emb_dim
    pretrain_w2v = args.pretrain_w2v
    max_length = args.max_length_document
    num_kernel_per_ws = args.num_kernel_per_ws

    if res_dir is None:
        sys.exit("Argument missing - res_dir is required")
    else:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    fixed_res_dir = os.path.join(res_dir, 'U_V', 'fold-1')
    if not os.path.exists(fixed_res_dir):
        os.makedirs(fixed_res_dir)

    print "===================================Options==================================="
    print "\taux path - %s" % aux_path
    print "\tdata path - %s" % data_path
    print "\tresult path - %s" % res_dir
    print "\tpretrained w2v data path - %s" % pretrain_w2v
    R, D_all = data_factory.load(aux_path)
    CNN_X = D_all['X_sequence']
    vocab_size = len(D_all['X_vocab']) + 1

    if pretrain_w2v is None:
        init_W = None
    else:
        init_W = data_factory.read_pretrained_word2vec(
            pretrain_w2v, D_all['X_vocab'], emb_dim)

    if os.path.exists(os.path.join(fixed_res_dir, 'score.npy')):
        os.remove(os.path.join(fixed_res_dir, 'score.npy'))

    all_avg_results = {}
    all_val_rmse = {}
    c = 1

    # num_folds = 5
    # for f in range(1,num_folds+1):
    fold = 1
    train_user = data_factory.read_rating(
        os.path.join(data_path, 'fold-{}'.format(fold), 'train-fold_{}-users.dat'.format(fold)))
    train_item = data_factory.read_rating(
        os.path.join(data_path, 'fold-{}'.format(fold), 'train-fold_{}-items.dat'.format(fold)))
    valid_user = data_factory.read_rating(
        os.path.join(data_path, 'fold-{}'.format(fold), 'validation-fold_{}-users.dat'.format(fold)))
    test_user = data_factory.read_rating(
        os.path.join(data_path, 'fold-{}'.format(fold), 'test-fold_{}-users.dat'.format(fold)))

    print('Fold paths:')
    print(os.path.join(data_path, 'fold-{}'.format(fold), 'train-fold_{}-users.dat'.format(fold)))
    print(os.path.join(data_path, 'fold-{}'.format(fold), 'train-fold_{}-items.dat'.format(fold)))
    print(os.path.join(data_path, 'fold-{}'.format(fold), 'validation-fold_{}-users.dat'.format(fold)))
    print(os.path.join(data_path, 'fold-{}'.format(fold), 'test-fold_{}-users.dat'.format(fold)))

    for lambda_u, lambda_v, confidence_mod, content_mode in itertools.product(lambda_u_list, lambda_v_list,
                                                                              confidence_mods, content_mods):
        experiment = '{}-{}-{}-{}'.format(lambda_u, lambda_v, confidence_mod, content_mode)
        if content_mode == 'cnn_cae':
            for att_dim in att_dims:
                experiment_cae = experiment + '-{}'.format(att_dim)
                experiment_dir = os.path.join(res_dir, experiment_cae)
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)
                print "==========================================================================================="
                print "## Hyperparameters for configuration setup %d out of %d \n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tconfidence_mod: %s" \
                      % (c, num_config, lambda_u, lambda_v, ('Constant' if confidence_mod == 'c' else 'user-dependent'))
                print "\tContent: %s" % ('Text and item attributes\n\tAttributes latent vector dim %d' % att_dim)
                print '\tJoin CNN and CAE outputs method: %s' % (
                    'Transfer block' if use_transfer_block else 'Concatenation')

                c += 1
                # Read item's attributes
                labels, features_matrix = data_factory.read_attributes(os.path.join(aux_path + 'paper_attributes.tsv'))
                if confidence_mod == 'c':
                    give_item_weight = False
                elif confidence_mod == 'w':
                    give_item_weight = True
                tr_eval, val_eval, te_eval = \
                    ConvCAEMF(max_iter=max_iter, res_dir=fixed_res_dir,
                              state_log_dir=os.path.join(experiment_dir, 'fold-{}'.format(fold)),
                              lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, vocab_size=vocab_size,
                              init_W=init_W,
                              give_item_weight=give_item_weight, CNN_X=CNN_X, emb_dim=emb_dim,
                              num_kernel_per_ws=num_kernel_per_ws,
                              train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user,
                              R=R,
                              attributes_X=features_matrix, att_dim=att_dim, use_transfer_block=use_transfer_block)

                # evaluator = Evaluator(R.shape[0], os.path.abspath(os.path.join(fixed_res_dir, os.pardir)))
                # if os.path.exists(os.path.join(fixed_res_dir, 'score.npy')):
                #     os.remove(os.path.join(fixed_res_dir, 'score.npy'))
                #
                # results = evaluator.eval_experiment(splits_dir)
                # avg_results = list(map(float, results[-1][1:]))
                # all_avg_results[experiment_cae] = avg_results
                # pickl.dump(results, open(os.path.join(experiment_dir, "metrics_matrix.dat"), "wb"))
                all_val_rmse[experiment_cae] = [tr_eval, val_eval, te_eval]

        elif content_mode == 'cnn':
            experiment_dir = os.path.join(res_dir, experiment)
            if not os.path.exists(experiment):
                os.makedirs(experiment)
            print "==========================================================================================="
            print "## Hyperparameters for configuration setup %d out of %d \n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tconfidence_mod: %s" \
                  % (c, num_config, lambda_u, lambda_v, ('Constant' if confidence_mod == 'c' else 'user-dependent'))
            # print "\tContent: %s" % ('Text and item attributes' if content_mode == 'cnn_cae' else 'Text')
            print "\tContent: %s" % 'Text'

            c += 1

            if confidence_mod == 'c':
                give_item_weight = False
            elif confidence_mod == 'w':
                give_item_weight = True

            tr_eval, val_eval, te_eval = \
                ConvMF(max_iter=max_iter, res_dir=fixed_res_dir,
                       state_log_dir=os.path.join(experiment_dir, 'fold-{}'.format(fold)),
                       lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, vocab_size=vocab_size, init_W=init_W,
                       give_item_weight=give_item_weight, CNN_X=CNN_X, emb_dim=emb_dim,
                       num_kernel_per_ws=num_kernel_per_ws,
                       train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)
            # evaluator = Evaluator(R.shape[0], os.path.abspath(os.path.join(fixed_res_dir, os.pardir)))
            # if os.path.exists(os.path.join(fixed_res_dir, 'score.npy')):
            #     os.remove(os.path.join(fixed_res_dir, 'score.npy'))
            # results = evaluator.eval_experiment(splits_dir)
            # avg_results = list(map(float, results[-1][1:]))
            # all_avg_results[experiment] = avg_results
            # pickl.dump(results, open(os.path.join(experiment_dir, "metrics_matrix.dat"), "wb"))

            # Stor rmse
            all_val_rmse[experiment] = [tr_eval, val_eval, te_eval]

        elif content_mode == 'cae':
            for att_dim in att_dims:
                experiment_cae = experiment + '-{}'.format(att_dim)
                experiment_dir = os.path.join(res_dir, experiment_cae)
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)
                print "==========================================================================================="
                print "## Hyperparameters for configuration setup %d out of %d \n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tconfidence_mod: %s" \
                      % (c, num_config, lambda_u, lambda_v,
                         ('Constant' if confidence_mod == 'c' else 'user-dependent'))
                print "\tContent: %s" % ('Attributes\n\tAttributes latent vector dim %d' % att_dim)

                c += 1
                # Read item's attributes
                labels, features_matrix = data_factory.read_attributes(
                    os.path.join(aux_path + 'paper_attributes.tsv'))

                if confidence_mod == 'c':
                    give_item_weight = False
                elif confidence_mod == 'w':
                    give_item_weight = True
                # attributes dimension must be equal to u, and v vectors dimension
                tr_eval, val_eval, te_eval = CAEMF(max_iter=max_iter, res_dir=fixed_res_dir,
                                                   state_log_dir=os.path.join(experiment_dir, 'fold-{}'.format(fold)),
                                                   lambda_u=lambda_u, lambda_v=lambda_v, dimension=att_dim,
                                                   att_dim=att_dim,
                                                   give_item_weight=give_item_weight,
                                                   train_user=train_user, train_item=train_item, valid_user=valid_user,
                                                   test_user=test_user, R=R,
                                                   attributes_X=features_matrix)

                # evaluator = Evaluator(R.shape[0], os.path.abspath(os.path.join(fixed_res_dir, os.pardir)))
                # if os.path.exists(os.path.join(fixed_res_dir, 'score.npy')):
                #     os.remove(os.path.join(fixed_res_dir, 'score.npy'))
                #
                # results = evaluator.eval_experiment(splits_dir)
                # avg_results = list(map(float, results[-1][1:]))
                # all_avg_results[experiment_cae] = avg_results
                # pickl.dump(results, open(os.path.join(experiment_dir, "metrics_matrix.dat"), "wb"))

                # Stor rmse
                all_val_rmse[experiment_cae] = [tr_eval, val_eval, te_eval]

        elif content_mode == 'mf':

            experiment_dir = os.path.join(res_dir, experiment)
            if not os.path.exists(experiment):
                os.makedirs(experiment)
            print "==========================================================================================="
            print "## Hyperparameters for configuration setup %d out of %d \n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tconfidence_mod: %s" \
                  % (c, num_config, lambda_u, lambda_v, ('Constant' if confidence_mod == 'c' else 'user-dependent'))
            # print "\tContent: %s" % ('Text and item attributes' if content_mode == 'cnn_cae' else 'Text')
            print "\tContent: %s" % 'Vanilla Matrix factorization'

            c += 1
            if confidence_mod == 'c':
                give_item_weight = False
            elif confidence_mod == 'w':
                give_item_weight = True
            tr_eval, val_eval, te_eval = \
                MF(max_iter=max_iter, res_dir=fixed_res_dir,
                   state_log_dir=os.path.join(experiment_dir, 'fold-{}'.format(fold)),
                   lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension,
                   give_item_weight=give_item_weight,
                   train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)

            # evaluator = Evaluator(R.shape[0], os.path.abspath(os.path.join(fixed_res_dir, os.pardir)))
            # if os.path.exists(os.path.join(fixed_res_dir, 'score.npy')):
            #     os.remove(os.path.join(fixed_res_dir, 'score.npy'))
            # results = evaluator.eval_experiment(splits_dir)
            # avg_results = list(map(float, results[-1][1:]))
            # all_avg_results[experiment] = avg_results
            # pickl.dump(results, open(os.path.join(experiment_dir, "metrics_matrix.dat"), "wb"))

            all_val_rmse[experiment] = [tr_eval, val_eval, te_eval]

    print 'Writing avg results for all sets of configuratoins to %s' % os.path.join(res_dir, 'all_avg_results_tanh.dat')
    pickl.dump(all_avg_results, open(os.path.join(res_dir, 'all_avg_results_tanh.dat'), "wb"))
    print 'Writing rmse for all sets of configuratoins to %s' % os.path.join(res_dir, 'all_rmse.dat')
    pickl.dump(all_val_rmse, open(os.path.join(res_dir, 'all_rmse.dat'), "wb"))
