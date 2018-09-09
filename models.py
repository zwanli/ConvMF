'''
Created on Dec 8, 2015

@author: donghyun
'''

import os
import time
import copy
from util import eval_RMSE
import math
import numpy as np
from text_analysis.models import CNN_CAE_module
from text_analysis.models import CNN_module
from text_analysis.models import CAE_module, CNN_CAE_transfer_module
from text_analysis.models import Stacking_NN_CNN_CAE
from lib.tensorboard_logging import Tb_Logger
from keras.callbacks import TensorBoard
import datetime


min_iter = 15
endure_count = 10

def get_rated_items_idx_map(train_R_I):
    '''

    :param train_R_I:
    :return: 1. list of rated items idx.
     2. A dict that maps the original item id, to a new one (it's the same in case of in-matrix splits,
    and a new one in the case of out-of-matrix splits.

    '''

    items_idx = set(np.concatenate(train_R_I).ravel().tolist())
    item_idx_to_new_id_map = {}
    for i,k in enumerate(items_idx):
        item_idx_to_new_id_map[i] = k
    return list(items_idx), item_idx_to_new_id_map


def map_theta_to_V(theta,id_to_original_map, m,k ):
    '''
    This function works as gather

    :param theta: (num_of_rated_items, embed_dim) array
    :return: (num_item, embed_dim) array. It's the same input theta in the case of in-matrix, and an expanded one with
    zeros rows in the case of out-of-matrix.
    '''
    expanded_theta = np.zeros((m,k),dtype=np.float)
    for i in range(theta.shape[0]):
        expanded_theta[id_to_original_map[i]]=theta[i]

    return expanded_theta


def ConvCAEMF(res_dir,state_log_dir, train_user, train_item, valid_user, test_user,
              R, attributes_X, CNN_X, vocab_size, init_W,att_dim,
              max_iter, lambda_u, lambda_v, dimension,
              dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100,
              a=1, b=0.01,  give_item_weight=False, use_transfer_block =True):
    # explicit setting
    # a = 1
    # b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]

    num_features = attributes_X.shape[1]

    '''prepare path to store results and log'''
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    os.chdir(res_dir)
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')

    '''log metrics using tf.summary '''
    log_dir_name = os.path.basename(os.path.dirname(state_log_dir+'/'))
    log_dir = os.path.join(state_log_dir,log_dir_name)
    logger_tb = Tb_Logger(log_dir)
    # indicate folder to save, plus other options
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=False, write_images=False)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]
    # then pass to fit as callback, remember to use validation_data also

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]

    # check if the dataset has validation set
    no_validation = False
    if valid_user:
        Valid_R = valid_user[1]
    else:
        no_validation = True

    #assign weights to each item according to the number of time the item was rated
    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        item_weight[item_weight == 0] = 1
    else:
        item_weight = np.ones(num_item, dtype=float)

    '''initialize'''
    if use_transfer_block:
        cnn_cae_module = CNN_CAE_transfer_module(dimension, vocab_size, dropout_rate,
                                    emb_dim, max_len, num_kernel_per_ws, init_W, cae_N_hidden=att_dim,
                                    nb_features=num_features)
    else:
        cnn_cae_module = CNN_CAE_module(dimension, vocab_size, dropout_rate,
                                    emb_dim, max_len, num_kernel_per_ws, init_W,cae_N_hidden=att_dim,
                                        nb_features=num_features)
    theta = cnn_cae_module.get_projection_layer(CNN_X, attributes_X)
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = theta

    print ('Training CNN-CAE-MF ...')
    pre_val_eval = -1e10
    PREV_LOSS = -1e-50
    endure_count = 5
    count = 0
    converge_threshold = 1e-4
    converge = 1.0
    iteration = 0
    while (iteration < max_iter and converge > converge_threshold) or iteration < min_iter:
        # for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        # Update U
        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in xrange(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        # Update V
        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            if len(U_j) > 0:
                tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
                A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
                B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                     ).sum(0) + lambda_v * item_weight[j] * theta[j]
                V[j] = np.linalg.solve(A, B)

                sub_loss[j] = -0.5 * np.square(R_j * a).sum()
                sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
                sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])
            else:
                #in case the item has no ratings
                V[j] = theta[j]
        loss = loss + np.sum(sub_loss)

        # Update theta
        seed = np.random.randint(100000)
        history = cnn_cae_module.train(CNN_X, V, att_train=attributes_X, item_weight=item_weight,
                                       seed=seed,callbacks_list=callbacks_list)
        theta = cnn_cae_module.get_projection_layer(CNN_X, attributes_X)
        cnn_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item

        toc = time.time()
        elapsed = toc - tic

        '''calculate RMSE'''
        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        if not no_validation:
            val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        else:
            val_eval = -1
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        ''' write tf.summary'''
        logger_tb.log_scalar('train_rmse',tr_eval,iteration)
        if not no_validation:
            logger_tb.log_scalar('eval_rmse',val_eval,iteration)
        logger_tb.log_scalar('test_rmse',te_eval,iteration)
        logger_tb.writer.flush()


        '''Calculate converge and stor best values of U,V,theta'''
        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        # if (val_eval < pre_val_eval):
        if (loss > PREV_LOSS):
            #count = 0
            print ("likelihood is increasing!")
            cnn_cae_module.save_model(res_dir + '/CNN_CAE_weights.hdf5')
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
            best_train_rmse = tr_eval
            best_test_rmse = te_eval
            best_val_rmse = val_eval

        else:
            count = count + 1

        pre_val_eval = val_eval

        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))
        if (count >= endure_count and iteration > min_iter):
        #if (count == endure_count):
            break
        elif (iteration < min_iter) :
            count = 0

        PREV_LOSS = loss
        iteration += 1
    f1.close()
    return best_train_rmse, best_test_rmse, best_val_rmse

def ConvMF(res_dir, state_log_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, init_W=None, give_item_weight=False,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    # explicit settinggit
    a = 1
    b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = -1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    os.chdir(res_dir)
    #f1 = open(res_dir + '/state.log', 'w')
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')
    # log metrics into tf.summary
    log_dir_name = os.path.basename(os.path.dirname(state_log_dir+'/'))
    log_dir = os.path.join(state_log_dir,log_dir_name)
    logger_tb = Tb_Logger(log_dir)

    # indicate folder to save, plus other options
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=False, write_images=False)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]
    # then pass to fit as callback, remember to use validation_data also

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]

    no_validation = False
    if valid_user:
        Valid_R = valid_user[1]
    else:
        no_validation = True

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        item_weight[item_weight == 0] = 1
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10

    cnn_module = CNN_module(dimension, vocab_size, dropout_rate,
                            emb_dim, max_len, num_kernel_per_ws, init_W)
    theta = cnn_module.get_projection_layer(CNN_X)
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = theta

    print ('Training CNN-MF ...')

    endure_count = 5
    count = 0
    converge_threshold = 1e-4
    converge = 1.0
    iteration = 0
    while (iteration < max_iter and converge > converge_threshold) or iteration < min_iter:
        # for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in xrange(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            if len(U_j) > 0:
                tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
                A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
                B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                     ).sum(0) + lambda_v * item_weight[j] * theta[j]
                V[j] = np.linalg.solve(A, B)

                sub_loss[j] = -0.5 * np.square(R_j * a).sum()
                sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
                sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])
            else:
                V[j] = theta[j]

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        history = cnn_module.train(CNN_X, V, item_weight, seed,callbacks_list)
        theta = cnn_module.get_projection_layer(CNN_X)
        cnn_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item


        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        if not no_validation:
            val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        else:
            val_eval = -1
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        logger_tb.log_scalar('train_rmse',tr_eval,iteration)
        if not no_validation:
            logger_tb.log_scalar('eval_rmse',val_eval,iteration)
        logger_tb.log_scalar('test_rmse',te_eval,iteration)
        logger_tb.writer.flush()

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        # if (val_eval < pre_val_eval):

        if (loss > PREV_LOSS):
            #count = 0

            print ("likelihood is increasing!")
            cnn_module.save_model(res_dir + '/CNN_weights.hdf5')
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
            best_train_rmse = tr_eval
            best_test_rmse = te_eval
            best_val_rmse = val_eval

        else:
            count = count + 1
        # if (val_eval < pre_val_eval):
        # count = 0

        #     cnn_module.save_model(res_dir + '/CNN_weights.hdf5')
        #     np.savetxt(res_dir + '/final-U.dat', U)
        #     np.savetxt(res_dir + '/final-V.dat', V)
        #     np.savetxt(res_dir + '/theta.dat', theta)
        # else:
        #     count = count + 1

        pre_val_eval = val_eval

        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))

        if (count >= endure_count and iteration > min_iter):
        #if (count == endure_count):
            break
        elif (iteration < min_iter) :
            count = 0

        PREV_LOSS = loss
        iteration += 1
    f1.close()
    return best_train_rmse, best_test_rmse, best_val_rmse

def CAEMF(res_dir,state_log_dir, train_user, train_item, valid_user, test_user,
              R, attributes_X, give_item_weight=False,
              max_iter=50, lambda_u=1, lambda_v=100, dimension=200,
              a=1, b=0.01, att_dim=50):
    # explicit setting
    # a = 1
    # b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]
    num_features = attributes_X.shape[1]
    PREV_LOSS = -1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    os.chdir(res_dir)
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')
    # log metrics into tf.summary
    log_dir_name = os.path.basename(os.path.dirname(state_log_dir+'/'))
    log_dir = os.path.join(state_log_dir,log_dir_name)
    logger_tb = Tb_Logger(log_dir)

    # indicate folder to save, plus other options
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=False, write_images=False)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]
    # then pass to fit as callback, remember to use validation_data also

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]

    no_validation = False
    if valid_user:
        Valid_R = valid_user[1]
    else:
        no_validation = True

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        item_weight[item_weight == 0] = 1
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10

    cae_module = CAE_module(dimension,cae_N_hidden=att_dim, nb_features=num_features)
    theta = cae_module.get_projection_layer(attributes_X)
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = theta

    endure_count = 5
    count = 0

    print ('Training CAE-MF ...')
    converge_threshold = 1e-4
    converge = 1.0
    iteration = 0
    while (iteration < max_iter and  converge > converge_threshold) or iteration < min_iter :
    # for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in xrange(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            if len(U_j) > 0:
                tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
                A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
                B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                     ).sum(0) + lambda_v * item_weight[j] * theta[j]
                V[j] = np.linalg.solve(A, B)

                sub_loss[j] = -0.5 * np.square(R_j * a).sum()
                sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
                sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])
            else:
                V[j]=theta[j]

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        history = cae_module.train(V, item_weight, seed, att_train=attributes_X,callbacks_list=callbacks_list)
        theta = cae_module.get_projection_layer(attributes_X)
        cae_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cae_loss * num_item

        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        if not no_validation:
            val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        else:
            val_eval = -1
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        logger_tb.log_scalar('train_rmse', tr_eval, iteration)
        if not no_validation:
            logger_tb.log_scalar('eval_rmse', val_eval, iteration)
        logger_tb.log_scalar('test_rmse', te_eval, iteration)
        logger_tb.writer.flush()

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        # if (val_eval < pre_val_eval):
        if (loss > PREV_LOSS):
            #count = 0

            print ("likelihood is increasing!")
            cae_module.save_model(res_dir + '/CAE_weights.hdf5')
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
            best_train_rmse = tr_eval
            best_test_rmse = te_eval
            best_val_rmse = val_eval

        else:
            count = count + 1

        # if (val_eval < pre_val_eval):
        #     count = 0
        #     cae_module.save_model(res_dir + '/CAE_weights.hdf5')
        #     np.savetxt(res_dir + '/final-U.dat', U)
        #     np.savetxt(res_dir + '/final-V.dat', V)
        #     np.savetxt(res_dir + '/theta.dat', theta)
        # else:
        #     count = count + 1

        pre_val_eval = val_eval

        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))

        if (count >= endure_count and iteration > min_iter):
        #if (count == endure_count):
            break
        elif (iteration < min_iter) :
            count = 0

        PREV_LOSS = loss
        iteration += 1

    f1.close()
    return best_train_rmse, best_test_rmse, best_val_rmse


def MF(res_dir,state_log_dir, train_user, train_item, valid_user, test_user,
              R, give_item_weight=False,
              max_iter=50, lambda_u=1, lambda_v=100, dimension=200,
              a=1, b=0.01):
    # explicit setting
    # a = 1
    # b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]

    PREV_LOSS = 1e50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    os.chdir(res_dir)
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')
    # log metrics into tf.summary
    log_dir_name = os.path.basename(os.path.dirname(state_log_dir+'/'))
    log_dir = os.path.join(state_log_dir,log_dir_name)
    logger_tb = Tb_Logger(log_dir)

    # indicate folder to save, plus other options
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=False, write_images=False)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]
    # then pass to fit as callback, remember to use validation_data also

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]

    no_validation = False
    if valid_user:
        Valid_R = valid_user[1]
    else:
        no_validation = True

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        item_weight[item_weight == 0] = 1
    else:
        item_weight = np.ones(num_item, dtype=float)



    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = np.random.uniform(size=(num_item, dimension))

    
    converge_threshold = 1e-4
    converge = 1.0
    pre_val_eval = 1e10
    PREV_LOSS = -1e-50

    count = 0


    print ('Training MF ...')
    iteration = 0
    while (iteration < max_iter and  converge > converge_threshold) or iteration < min_iter :
    # for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in xrange(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                 ).sum(0)
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)


        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        if not no_validation:
            val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        else:
            val_eval = -1
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        logger_tb.log_scalar('train_rmse', tr_eval, iteration)
        if not no_validation:
            logger_tb.log_scalar('evale_rmse', val_eval, iteration)
        logger_tb.log_scalar('test_rmse', te_eval, iteration)
        logger_tb.writer.flush()

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)


        if (loss > PREV_LOSS):
        # if (val_eval < pre_val_eval):
            #count = 0
            print ("likelihood is increasing!")
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            best_train_rmse = tr_eval
            best_test_rmse = te_eval
            best_val_rmse = val_eval
        else:
            count = count + 1

        # if (val_eval < pre_val_eval):
        #     count = 0
        #     np.savetxt(res_dir + '/final-U.dat', U)
        #     np.savetxt(res_dir + '/final-V.dat', V)
        # else:
        #     count = count + 1

        pre_val_eval = val_eval

        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))

        if (count == endure_count):
            break

        PREV_LOSS = loss
        iteration += 1
    f1.close()

    return best_train_rmse, best_test_rmse, best_val_rmse

def stacking_CNN_CAE(res_dir,state_log_dir, train_user, train_item, valid_user, test_user,
              R, CNN_theta, CAE_gamma,
              max_iter, lambda_u, lambda_v, dimension, lr = 0.01,
              dropout_rate=0.2, a=1, b=0.01,  give_item_weight=False):
    # explicit setting
    # a = 1
    # b = 0.01
    num_user = R.shape[0]
    num_item = R.shape[1]
    '''prepare path to store results and log'''
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    os.chdir(res_dir)
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')

    '''log metrics using tf.summary '''
    log_dir_name = os.path.basename(os.path.dirname(state_log_dir+'/'))
    log_dir = os.path.join(state_log_dir,log_dir_name)
    logger_tb = Tb_Logger(log_dir)
    # indicate folder to save, plus other options
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=False, write_images=False)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]
    # then pass to fit as callback, remember to use validation_data also

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]

    # check if the dataset has validation set
    no_validation = False
    if valid_user:
        Valid_R = valid_user[1]
    else:
        no_validation = True

    #assign weights to each item according to the number of time the item was rated
    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        item_weight[item_weight == 0] = 1
    else:
        item_weight = np.ones(num_item, dtype=float)

    alpha = 1
    beta = 1
    # theta = (alpha* CNN_theta + beta *CAE_gamma) / 2
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = (alpha* CNN_theta + beta *CAE_gamma) / 2

    print ('Training Stacking-CNN-CAE_MF ...')
    pre_val_eval = -1e10
    PREV_LOSS = -1e-50
    endure_count = 5
    count = 0
    converge_threshold = 1e-4
    converge = 1.0
    iteration = 0
    while (iteration < max_iter and converge > converge_threshold) or iteration < min_iter:
        # for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        # Update U
        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in xrange(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        # Update V
        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            if len(U_j) > 0:
                tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
                A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
                B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                     ).sum(0) + lambda_v * item_weight[j] * (alpha* CNN_theta[j] + beta *CAE_gamma[j])
                V[j] = np.linalg.solve(A, B)

                sub_loss[j] = -0.5 * np.square(R_j * a).sum()
                sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
                sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])
            else:
                #in case the item has no ratings
                V[j] = (alpha* CNN_theta[j] + beta *CAE_gamma[j])
        loss = loss + np.sum(sub_loss)

        # Update alpha and beta
        seed = np.random.randint(100000)
        alpha = alpha - lr * (np.sum(CNN_theta*(V - (alpha* CNN_theta + beta *CAE_gamma))))
        alpha_loss =- np.sum((V - (alpha* CNN_theta + beta *CAE_gamma))**2)
        beta = beta - lr * (np.sum(CAE_gamma*(V - (alpha* CNN_theta + beta *CAE_gamma))))
        beta_loss =np.sum((V - (alpha* CNN_theta + beta *CAE_gamma))**2)

        loss = loss - 0.5 * lambda_v * (alpha_loss+beta_loss)

        toc = time.time()
        elapsed = toc - tic

        '''calculate RMSE'''
        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        if not no_validation:
            val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        else:
            val_eval = -1
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        ''' write tf.summary'''
        logger_tb.log_scalar('train_rmse',tr_eval,iteration)
        if not no_validation:
            logger_tb.log_scalar('eval_rmse',val_eval,iteration)
        logger_tb.log_scalar('test_rmse',te_eval,iteration)
        logger_tb.writer.flush()


        '''Calculate converge and stor best values of U,V,theta'''
        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        # if (val_eval < pre_val_eval):
        if (loss > PREV_LOSS):
            #count = 0
            print ("likelihood is increasing!")
            with open(res_dir + '/alpha_beta.txt', 'w') as the_file:
                the_file.write('alpha: %f' % alpha)
                the_file.write('beta: %f' % beta)
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            best_train_rmse = tr_eval
            best_test_rmse = te_eval
            best_val_rmse = val_eval
        else:
            count = count + 1
        pre_val_eval = val_eval
        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))
        if (count >= endure_count and iteration > min_iter):
        #if (count == endure_count):
            break
        elif (iteration < min_iter) :
            count = 0

        PREV_LOSS = loss
        iteration += 1
    f1.close()
    return best_train_rmse, best_test_rmse, best_val_rmse

def NN_stacking_CNN_CAE(res_dir,state_log_dir, train_user, train_item, valid_user, test_user,
              R, CNN_theta, CAE_gamma,
              max_iter, lambda_u, lambda_v, dimension, lr = 0.01,
              dropout_rate=0.2, a=1, b=0.01,  give_item_weight=False):
    # explicit settinggit
    a = 1
    b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = -1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    os.chdir(res_dir)
    #f1 = open(res_dir + '/state.log', 'w')
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')
    # log metrics into tf.summary
    log_dir_name = os.path.basename(os.path.dirname(state_log_dir+'/'))
    log_dir = os.path.join(state_log_dir,log_dir_name)
    logger_tb = Tb_Logger(log_dir)

    # indicate folder to save, plus other options
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=False, write_images=False)
    # save it in your callback list, where you can include other callbacks
    callbacks_list = [tensorboard]
    # then pass to fit as callback, remember to use validation_data also

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]

    no_validation = False
    if valid_user:
        Valid_R = valid_user[1]
    else:
        no_validation = True

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        item_weight[item_weight == 0] = 1
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10

    theta_gamma_train = np.concatenate([CNN_theta,CAE_gamma],axis=1)
    stacking_module = Stacking_NN_CNN_CAE(dimension, theta_gamma_train.shape[1],dropout_rate=dropout_rate)
    phi = stacking_module.get_projection_layer(theta_gamma_train)
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    if (theta_gamma_train.shape == dimension*2):
        V = (CNN_theta +CAE_gamma) / 2
    else:
        V = CNN_theta

    print ('Training CNN-MF ...')

    endure_count = 5
    count = 0
    converge_threshold = 1e-4
    converge = 1.0
    iteration = 0
    while (iteration < max_iter and converge > converge_threshold) or iteration < min_iter:
        # for iteration in xrange(max_iter):
        loss = 0
        tic = time.time()
        print "%d iteration\t(patience: %d)" % (iteration, count)

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in xrange(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in xrange(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            if len(U_j) > 0:
                tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
                A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
                B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                     ).sum(0) + lambda_v * item_weight[j] * phi[j]
                V[j] = np.linalg.solve(A, B)

                sub_loss[j] = -0.5 * np.square(R_j * a).sum()
                sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
                sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])
            else:
                V[j] = phi[j]

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        history = stacking_module.train(theta_gamma_train, V, item_weight, seed,callbacks_list)
        theta = stacking_module.get_projection_layer(theta_gamma_train)
        stacking_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * stacking_loss * num_item


        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        if not no_validation:
            val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        else:
            val_eval = -1
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        logger_tb.log_scalar('train_rmse',tr_eval,iteration)
        if not no_validation:
            logger_tb.log_scalar('eval_rmse',val_eval,iteration)
        logger_tb.log_scalar('test_rmse',te_eval,iteration)
        logger_tb.writer.flush()

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        # if (val_eval < pre_val_eval):

        if (loss > PREV_LOSS):
            #count = 0

            print ("likelihood is increasing!")
            stacking_module.save_model(res_dir + '/stacking_weights.hdf5')
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
            best_train_rmse = tr_eval
            best_test_rmse = te_eval
            best_val_rmse = val_eval

        else:
            count = count + 1
        # if (val_eval < pre_val_eval):
        # count = 0

        #     cnn_module.save_model(res_dir + '/CNN_weights.hdf5')
        #     np.savetxt(res_dir + '/final-U.dat', U)
        #     np.savetxt(res_dir + '/final-V.dat', V)
        #     np.savetxt(res_dir + '/theta.dat', theta)
        # else:
        #     count = count + 1

        pre_val_eval = val_eval

        print "Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval)
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))

        if (count >= endure_count and iteration > min_iter):
        #if (count == endure_count):
            break
        elif (iteration < min_iter) :
            count = 0

        PREV_LOSS = loss
        iteration += 1
    f1.close()
    return best_train_rmse, best_test_rmse, best_val_rmse