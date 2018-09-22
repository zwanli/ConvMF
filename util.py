'''
Created on Dec 8, 2015

@author: donghyun
'''
import numpy as np


def eval_RMSE(R, U, V, TS):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    TS_count = 0
    for i in xrange(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        R_i = R[i]

        sub_rmse[i] = np.square(approx_R_i - R_i).sum()

    rmse = np.sqrt(sub_rmse.sum() / TS_count)

    return rmse


def make_CDL_format(X_base, path):
    max_X = X_base.max(1).toarray()
    for i in xrange(max_X.shape[0]):
        if max_X[i, 0] == 0:
            max_X[i, 0] = 1
    max_X_rep = np.tile(max_X, (1, X_base.shape[1]))
    X_nor = X_base / max_X_rep
    np.savetxt(path + '/mult_nor.dat', X_nor, fmt='%.5f')


def get_confidence_matrix(ratings, mode, **kwargs):
    '''
    :param mode: { 'constant','user-dependant' }
    :return:
    '''
    confidence_matrix = np.zeros((ratings.shape))
    if mode == 'constant':
        if 'alpha' in kwargs and 'beta' in kwargs:
            confidence_matrix[ratings == 1] = kwargs['alpha']
            confidence_matrix[ratings != 1] = kwargs['beta']
        else:
            raise Exception('alpha and beta values are required, where alpha >> beta ')
    elif mode == 'user-dependant':
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 40
            print('alpha value is not provided, using default value %d ' % alpha)
        count_nonzero = np.count_nonzero(ratings, axis=1)
        confidence_matrix = confidence_matrix.T + (1 + alpha * count_nonzero)
        confidence_matrix = confidence_matrix.T
    else:
        print('Using default confidence mode, constant  ')
    return confidence_matrix


def print_helper(content_type):
    '''
    A helper funtction that returns a longer description of the content type
    :param content_type:
    :return:
    '''
    if content_type == 'cnn_cae':
        return 'Text and attributes'
    elif content_type == 'cnn':
        return 'Text'
    elif content_type == 'cae':
        return 'Attributes'
    elif content_type == 'stacking':
        return 'Stacking ensemble'
    elif content_type == 'nn_stacking':
        return 'NN stacking ensebmle'
    elif content_type == 'mf':
        return 'Vanilla matrix factorization'
    elif content_type =='raw_att_cnn':
        return 'FC( Raw attributes), and CNN trained separately '
    else:
        return 'Content mode parser failed'

import sys

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
