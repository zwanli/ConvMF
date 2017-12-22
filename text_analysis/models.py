'''
Created on Dec 8, 2015
@author: donghyun

modified by: @atlas90

contractive_autoencoder by @wiseodd

'''

import numpy as np

np.random.seed(1337)

from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Reshape, Flatten, Dropout
from keras.layers import Input, Embedding, Dense, concatenate
from keras.models import Model, Sequential
from keras.preprocessing import sequence
import keras.backend as K


# from keras.utils import plot_model


class CNN_CAE_module():
    '''
    classdocs
    '''
    batch_size = 256
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters,
                 init_W=None, cae_N_hidden=50, nb_features=17):

        ''' CNN Module'''
        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]
        # self.model = Graph()
        # input
        doc_input = Input(shape=(max_len,), dtype='int32', name='doc_input')

        '''Embedding Layer'''
        if init_W is None:
            # self.model.add_node(Embedding(
            #     max_features, emb_dim, input_length=max_len), name='sentence_embeddings', input='input')
            sentence_embeddings = Embedding(output_dim=emb_dim, input_dim=max_features, input_length=max_len,
                                            name='sentence_embeddings')(doc_input)
        else:
            # self.model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=[
            #                     init_W / 20]), name='sentence_embeddings', input='input')
            sentence_embeddings = Embedding(output_dim=emb_dim, input_dim=max_features, input_length=max_len,
                                            weights=[init_W / 20], name='sentence_embeddings')(doc_input)

        '''Reshape Layer'''
        reshape = Reshape(target_shape=(max_len, emb_dim, 1), name='reshape')(sentence_embeddings)  # chanels last

        '''Convolution Layer & Max Pooling Layer'''
        flatten_ = []
        for i in filter_lengths:
            model_internal = Sequential()
            # model_internal.add(Convolution2D(
            #     nb_filters, i, emb_dim, activation="relu"))
            model_internal.add(Conv2D(nb_filters, (i, emb_dim), activation="relu",
                                      name='conv2d_' + str(i), input_shape=(self.max_len, emb_dim, 1)))
            # model_internal.add(MaxPooling2D(
            #     pool_size=(self.max_len - i + 1, 1)))
            model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1), name='maxpool2d_' + str(i)))
            model_internal.add(Flatten())
            flatten = model_internal(reshape)
            flatten_.append(flatten)

        '''Fully Connect Layer & Dropout Layer'''
        # self.model.add_node(Dense(vanila_dimension, activation='tanh'),
        #                     name='fully_connect', inputs=['unit_' + str(i) for i in filter_lengths])
        fully_connect = Dense(vanila_dimension, activation='tanh',
                              name='fully_connect')(concatenate(flatten_, axis=-1))

        # self.model.add_node(Dropout(dropout_rate),
        #                     name='dropout', input='fully_connect')
        dropout = Dropout(dropout_rate, name='dropout')(fully_connect)

        ''' Attributes module '''
        lam = 1e-3
        N = nb_features
        # cae_N_hidden = 50

        att_input = Input(shape=(N,), name='cae_input')
        encoded = Dense(cae_N_hidden, activation='sigmoid', name='encoded')(att_input)
        att_output = Dense(N, activation='linear', name='cae_output')(encoded)

        # model = Model(input=att_input, output=att_output)

        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)

            W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x cae_N_hidden
            W = K.transpose(W)  # cae_N_hidden x N
            h = model.get_layer('encoded').output
            dh = h * (1 - h)  # N_batch x cae_N_hidden

            # N_batch x cae_N_hidden * cae_N_hidden x 1 = N_batch x 1
            contractive = lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

            return mse + contractive

        joint_output = concatenate([dropout, encoded])

        '''Projection Layer & Output Layer'''
        # self.model.add_node(Dense(projection_dimension, activation='tanh'),
        #                     name='projection', input='dropout')
        pj = Dense(projection_dimension, activation='tanh', name='joint_output')  # output layer
        projection = pj(joint_output)

        # Output Layergit
        model = Model(inputs=[doc_input, att_input], outputs=[projection, att_output])
        model.compile(optimizer='rmsprop',
                      # optimizer={'joint_output': 'rmsprop', 'cae_output':  'adam'},
                      loss={'joint_output': 'mse', 'cae_output': contractive_loss},
                      loss_weights={'joint_output': 1., 'cae_output': 1.})
        # plot_model(model, to_file='model.png')

        self.model = model

    def contractive_autoencoder(self, X, lam=1e-3):
        X = X.reshape(X.shape[0], -1)
        M, N = X.shape
        N_hidden = 64
        N_batch = 100

        inputs = Input(shape=(N,))
        encoded = Dense(N_hidden, activation='sigmoid', name='encoded')(inputs)
        outputs = Dense(N, activation='linear')(encoded)
        model = Model(input=inputs, output=outputs)

        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)

            W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
            W = K.transpose(W)  # N_hidden x N
            h = model.get_layer('encoded').output
            dh = h * (1 - h)  # N_batch x N_hidden

            # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
            contractive = lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

            return mse + contractive

        model.compile(optimizer='adam', loss=contractive_loss)
        model.fit(X, X, batch_size=N_batch, nb_epoch=3)

        return model, Model(input=inputs, output=encoded)

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
        self.max_len = max_len
        max_features = vocab_size

        filter_lengths = [3, 4, 5]
        print("Build model...")
        self.qual_model = Graph()
        self.qual_conv_set = {}
        '''Embedding Layer'''
        self.qual_model.add_input(
            name='input', input_shape=(max_len,), dtype=int)

        self.qual_model.add_node(Embedding(max_features, emb_dim, input_length=max_len,
                                           weights=self.model.nodes['sentence_embeddings'].get_weights()),
                                 name='sentence_embeddings', input='input')

        '''Convolution Layer & Max Pooling Layer'''
        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(
                Reshape(dims=(1, max_len, emb_dim), input_shape=(max_len, emb_dim)))
            self.qual_conv_set[i] = Convolution2D(nb_filters, i, emb_dim, activation="relu", weights=self.model.nodes[
                'unit_' + str(i)].layers[1].get_weights())
            model_internal.add(self.qual_conv_set[i])
            model_internal.add(MaxPooling2D(pool_size=(max_len - i + 1, 1)))
            model_internal.add(Flatten())

            self.qual_model.add_node(
                model_internal, name='unit_' + str(i), input='sentence_embeddings')
            self.qual_model.add_output(
                name='output_' + str(i), input='unit_' + str(i))

        self.qual_model.compile(
            'rmsprop', {'output_3': 'mse', 'output_4': 'mse', 'output_5': 'mse'})

    def train(self, X_train, V, item_weight, seed, att_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)

        np.random.seed(seed)
        V = np.random.permutation(V)

        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        np.random.seed(seed)
        att_train = np.random.permutation(att_train)

        print("Train...CNN_CAE module")
        history = self.model.fit({'doc_input': X_train, 'cae_input': att_train},
                                 {'joint_output': V, 'cae_output': att_train},
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight={'joint_output': item_weight})

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train, att_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        # Y = self.model.predict(
        #     {'doc_input': X_train, 'cae_input':att_train}, batch_size=len(X_train))
        Y = self.model.predict(
            {'doc_input': X_train, 'cae_input': att_train}, batch_size=2048)
        return Y[0]


class CNN_module():
    '''
    classdocs
    '''
    batch_size = 64
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]
        # self.model = Graph()
        # input
        doc_input = Input(shape=(max_len,), dtype='int32', name='doc_input')

        '''Embedding Layer'''

        if init_W is None:
            # self.model.add_node(Embedding(
            #     max_features, emb_dim, input_length=max_len), name='sentence_embeddings', input='input')
            sentence_embeddings = Embedding(output_dim=emb_dim, input_dim=max_features, input_length=max_len,
                                            name='sentence_embeddings')(doc_input)
        else:
            # self.model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=[
            #                     init_W / 20]), name='sentence_embeddings', input='input')
            sentence_embeddings = Embedding(output_dim=emb_dim, input_dim=max_features, input_length=max_len,
                                            weights=[init_W / 20], name='sentence_embeddings')(doc_input)

        '''Reshape Layer'''
        reshape = Reshape(target_shape=(max_len, emb_dim, 1), name='reshape')(sentence_embeddings)  # chanels last

        '''Convolution Layer & Max Pooling Layer'''
        flatten_ = []
        for i in filter_lengths:
            model_internal = Sequential()
            # model_internal.add(Convolution2D(
            #     nb_filters, i, emb_dim, activation="relu"))
            model_internal.add(Conv2D(nb_filters, (i, emb_dim), activation="relu",
                                      name='conv2d_' + str(i), input_shape=(self.max_len, emb_dim, 1)))
            # model_internal.add(MaxPooling2D(
            #     pool_size=(self.max_len - i + 1, 1)))
            model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1), name='maxpool2d_' + str(i)))
            model_internal.add(Flatten())
            flatten = model_internal(reshape)
            flatten_.append(flatten)

        '''Fully Connect Layer & Dropout Layer'''
        # self.model.add_node(Dense(vanila_dimension, activation='tanh'),
        #                     name='fully_connect', inputs=['unit_' + str(i) for i in filter_lengths])
        fully_connect = Dense(vanila_dimension, activation='tanh',
                              name='fully_connect')(concatenate(flatten_, axis=-1))

        # self.model.add_node(Dropout(dropout_rate),
        #                     name='dropout', input='fully_connect')
        dropout = Dropout(dropout_rate, name='dropout')(fully_connect)
        '''Projection Layer & Output Layer'''
        # self.model.add_node(Dense(projection_dimension, activation='tanh'),
        #                     name='projection', input='dropout')
        pj = Dense(projection_dimension, activation='tanh', name='output')  # output layer
        projection = pj(dropout)

        # Output Layergit
        model = Model(inputs=doc_input, outputs=projection)
        model.compile(optimizer='rmsprop', loss='mse')
        self.model = model

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(X_train, V,
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight={'output': item_weight})

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        # Y = self.model.predict(
        #     {'doc_input': X_train}, batch_size=len(X_train))
        Y = self.model.predict(
            {'doc_input': X_train}, batch_size=2048)
        return Y
