'''
Created on Dec 8, 2015

@author: donghyun
'''

import os
import time

from util import eval_RMSE
import math
import numpy as np
from text_analysis.models import CNN_CAE_module
from text_analysis.models import CNN_module
from text_analysis.models import CAE_module


def ConvCAEMF(res_dir,state_log_dir, train_user, train_item, valid_user, test_user,
              R, attributes_X, CNN_X, vocab_size, init_W=None, give_item_weight=False,
              max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
              dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100,
              a=1, b=0.01, att_dim=50):
    # explicit setting
    # a = 1
    # b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]
    num_features = attributes_X.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10

    cnn_cae_module = CNN_CAE_module(dimension, vocab_size, dropout_rate,
                                emb_dim, max_len, num_kernel_per_ws, init_W,cae_N_hidden=att_dim, nb_features=num_features)
    theta = cnn_cae_module.get_projection_layer(CNN_X, attributes_X)
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = theta

    print ('Training CNN-CAE-MF ...')
    endure_count = 5
    count = 0
    converge_threshold = 1e-4
    converge = 1.0
    iteration = 0
    while iteration < max_iter and converge > converge_threshold:
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
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        history = cnn_cae_module.train(CNN_X, V, item_weight, seed, att_train=attributes_X)
        theta = cnn_cae_module.get_projection_layer(CNN_X, attributes_X)
        cnn_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item

        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        if (loss > PREV_LOSS):
            count = 0
            print ("likelihood is increasing!")
            cnn_cae_module.save_model(res_dir + '/CNN_CAE_weights.hdf5')
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)

        else:
            count = count + 1
        # if (val_eval < pre_val_eval):
        # count = 0

        #     cnn_cae_module.save_model(res_dir + '/CNN_CAE_weights.hdf5')
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

        if (count == endure_count):
            break

        PREV_LOSS = loss
        iteration += 1
    f1.close()
    o = cnn_cae_module.get_intermediate_output(CNN_X, attributes_X)
    np.savetxt('cnn_cae_tanh.csv', o, fmt='%1.4f',delimiter=',')
    return tr_eval, val_eval, te_eval

def ConvMF(res_dir, state_log_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, init_W=None, give_item_weight=False,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    # explicit setting
    a = 1
    b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    #f1 = open(res_dir + '/state.log', 'w')
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
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
    while iteration < max_iter and converge > converge_threshold:
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
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        history = cnn_module.train(CNN_X, V, item_weight, seed)
        theta = cnn_module.get_projection_layer(CNN_X)
        cnn_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item

        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        if (loss > PREV_LOSS):
            count = 0

            print ("likelihood is increasing!")
            cnn_module.save_model(res_dir + '/CNN_weights.hdf5')
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)

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

        if (count == endure_count):
            break

        PREV_LOSS = loss
        iteration += 1
    f1.close()
    return tr_eval, val_eval, te_eval

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
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
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
    while iteration < max_iter and  converge > converge_threshold :
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
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        history = cae_module.train(V, item_weight, seed, att_train=attributes_X)
        theta = cae_module.get_projection_layer(attributes_X)
        cae_loss = history.history['loss'][-1]

        loss = loss - 0.5 * lambda_v * cae_loss * num_item

        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)


        if (loss > PREV_LOSS):
            count = 0

            print ("likelihood is increasing!")
            cae_module.save_model(res_dir + '/CAE_weights.hdf5')
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)

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

        if (count == endure_count):
            break

        PREV_LOSS = loss
        iteration += 1

    f1.close()
    return tr_eval, val_eval, te_eval


def MF(res_dir,state_log_dir, train_user, train_item, valid_user, test_user,
              R, give_item_weight=False,
              max_iter=50, lambda_u=1, lambda_v=100, dimension=200,
              a=1, b=0.01):
    # explicit setting
    # a = 1
    # b = 0.01

    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists(state_log_dir):
        os.makedirs(state_log_dir)
    f1 = open(state_log_dir + '/state.log', 'w')

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10

    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = np.random.uniform(size=(num_item, dimension))

    endure_count = 5
    count = 0

    converge_threshold = 1e-4
    converge = 1.0
    print ('Training MF ...')
    iteration = 0
    while iteration < max_iter and  converge > converge_threshold :
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
        val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)


        if (loss > PREV_LOSS):
            count = 0

            print ("likelihood is increasing!")
            np.savetxt(res_dir + '/final-U.dat', U)
            np.savetxt(res_dir + '/final-V.dat', V)
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

    return tr_eval, val_eval, te_eval
