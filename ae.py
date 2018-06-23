'''
Author @wiseodd
'''
from mpmath.functions.functions import im

from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Deconv2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from data_manager import Data_Factory
from util import Logger
# tf.python.control_flow_ops = tf


# mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
#
# X_train, y_train = mnist.train.images, mnist.train.labels
# X_val, y_val = mnist.validation.images, mnist.validation.labels
# X_test, y_test = mnist.test.images, mnist.test.labels
#
#
# def autoencoder(X, loss='l2', lam=0.):
#     X = X.reshape(X.shape[0], -1)
#     M, N = X.shape
#
#     inputs = Input(shape=(N,))
#     h = Dense(64, activation='sigmoid')(inputs)
#     outputs = Dense(N)(h)
#
#     model = Model(input=inputs, output=outputs)
#     loss = 'mae' if loss == 'l1' else 'mse'
#
#     model.compile(optimizer='adam', loss=loss)
#     model.fit(X, X, batch_size=64, nb_epoch=3)
#
#     return model, Model(input=inputs, output=h)
#
#
# def sparse_autoencoder(X, lam=1e-5):
#     X = X.reshape(X.shape[0], -1)
#     M, N = X.shape
#
#     inputs = Input(shape=(N,))
#     h = Dense(64, activation='sigmoid', activity_regularizer=activity_l1(lam))(inputs)
#     outputs = Dense(N)(h)
#
#     model = Model(input=inputs, output=outputs)
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, X, batch_size=64, nb_epoch=3)
#
#     return model, Model(input=inputs, output=h)
#
#
# def multilayer_autoencoder(X, lam=1e-5):
#     X = X.reshape(X.shape[0], -1)
#     M, N = X.shape
#
#     inputs = Input(shape=(N,))
#     h = Dense(128, activation='relu')(inputs)
#     encoded = Dense(64, activation='relu', activity_regularizer=activity_l1(lam))(h)
#     h = Dense(128, activation='relu')(encoded)
#     outputs = Dense(N)(h)
#
#     model = Model(input=inputs, output=outputs)
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, X, batch_size=64, nb_epoch=3)
#
#     return model, Model(input=inputs, output=h)
#
#
# def conv_autoencoder(X):
#     X = X.reshape(X.shape[0], 28, 28, 1)
#
#     inputs = Input(shape=(28, 28, 1))
#     h = Conv2D(4, 3, 3, activation='relu', border_mode='same')(inputs)
#     encoded = MaxPooling2D((2, 2))(h)
#     h = Conv2D(4, 3, 3, activation='relu', border_mode='same')(encoded)
#     h = UpSampling2D((2, 2))(h)
#     outputs = Conv2D(1, 3, 3, activation='relu', border_mode='same')(h)
#
#     model = Model(input=inputs, output=outputs)
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, X, batch_size=64, nb_epoch=5)
#
#     return model, Model(input=inputs, output=encoded)


def contractive_autoencoder(X, lam=1e-3):
    X = X.reshape(X.shape[0], -1)
    M, N = X.shape
    N_hidden = 64
    N_batch = 100

    inputs = Input(shape=(N,))
    encoded = Dense(N_hidden, activation='sigmoid', name='encoded')(inputs)
    outputs = Dense(N, activation='linear')(encoded)

    model = Model(inputs=inputs, outputs=outputs)

    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive

    model.compile(optimizer='rmsprop', loss=contractive_loss)
    model.fit(X, X, batch_size=N_batch, epochs=5)

    return model, Model(inputs=inputs, outputs=encoded)


if __name__ == '__main__':
    # data_factory = Data_Factory()
    # labels, X_train = data_factory.read_attributes('/home/wanli/data/Extended_ctr/convmf/dummy/preprocessed/paper_info_processed.csv')
    # model, representation = contractive_autoencoder(X_train)
    #
    # idx = [0,1,12,13,4]
    # X_recons = model.predict(X_train[idx])
    #
    # # idxs = np.random.randint(0, X_test.shape[0], size=5)
    # # X_recons = model.predict(X_test[idxs])
    #
    # for X_recon in X_recons:
    #     plt.imshow(X_recon.reshape(28, 28), cmap='Greys_r')
    #     plt.show()

    import cPickle as pickl
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    path = '/home/zaher/data/Extended_ctr/convmf/dummy/results'
    path = '/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/grid_search'
    # path = '/home/zaher/data/Extended_ctr/convmf/citeulike_a_extended/results/grid_cae'
    R = pickl.load(open(path + "/all_avg_results_tanh.dat", "rb"))
    recall_breaks = [5, 10] + list(xrange(20, 201, 20))
    mrr_breaks = [10]
    ndcg_breaks = [5, 10]
    results_header = ["Rec@" + str(i) for i in recall_breaks] + ["MRR@" + str(i) for i in mrr_breaks] + [
        "nDCG@" + str(i) for i in ndcg_breaks]
    print results_header
    df = pd.DataFrame.from_records(R.values(), index=R.keys(), columns=results_header)
    # df = df.cumsum()
    plt.figure()
    df.plot()
    # plt.show()
    df2 = df.loc[:, 'MRR@10':]
    df2 = df2.sort_values(by=['nDCG@5', 'nDCG@10', 'MRR@10'])
    df2.sort_index(inplace=True)

    df3 = df2.idxmax(axis=0, skipna=True)
    print(df3)
    print df