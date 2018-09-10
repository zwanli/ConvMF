'''
Created on Dec 8, 2015
@author: donghyun

modified by: @atlas90

contractive_autoencoder by @wiseodd

'''

import numpy as np
import sys

np.random.seed(1337)

from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.core import Reshape, Flatten, Dropout, Activation
from keras.layers import Input, Embedding, Dense, concatenate, BatchNormalization, add
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.utils import plot_model

from keras import backend as K
import tensorflow as tf

###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.8

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

batch_size = 128


class CNN_CAE_module():
    '''
    classdocs
    '''
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5
    batch_size = batch_size

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters,
                 init_W=None, cae_N_hidden=50, nb_features=17):

        ''' CNN Module'''
        model_summary = open('model_summary', 'w')

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
            # model_internal.add(Conv2D(
            #     nb_filters, i, emb_dim, activation="relu"))
            model_internal.add(Conv2D(nb_filters, (i, emb_dim), activation="relu",
                                      name='conv2d_' + str(i), input_shape=(self.max_len, emb_dim, 1)))
            # model_internal.add(MaxPooling2D(
            #     pool_size=(self.max_len - i + 1, 1)))
            model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1), name='maxpool2d_' + str(i)))
            model_internal.add(Flatten())
            flatten = model_internal(reshape)
            flatten_.append(flatten)
            model_internal.summary(print_fn=lambda x: model_summary.write(x + '\n'))

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

        # combine the outputs of boths modules

        joint_output = concatenate([dropout, encoded], name='concatenated_output')

        '''Projection Layer & Output Layer'''
        # self.model.add_node(Dense(projection_dimension, activation='tanh'),
        #                     name='projection', input='dropout')
        pj = Dense(projection_dimension, activation='tanh', name='joint_output')  # output layer
        projection = pj(joint_output)

        # Output Layergit
        model = Model(inputs=[doc_input, att_input], outputs=[projection, att_output])
        # todo: check the optimizer
        model.compile(optimizer='rmsprop',
                      # optimizer={'joint_output': 'rmsprop', 'cae_output':  'adam'},
                      loss={'joint_output': 'mse', 'cae_output': contractive_loss},
                      loss_weights={'joint_output': 1., 'cae_output': 1.})
        # plot_model(model, to_file='model.png')
        model.summary(print_fn=lambda x: model_summary.write(x + '\n'))

        self.model = model
        # plot_model(model, to_file='model_cnn_cae_concat2.png')

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

    def train(self, X_train, V, item_weight, seed, att_train, callbacks_list):
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
                                 sample_weight={'joint_output': item_weight}, callbacks=callbacks_list)

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

    def get_intermediate_output(self, X_train, att_train):
        layer_name = 'concatenated_output'
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer_name).output)
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        intermediate_output = intermediate_layer_model.predict({'doc_input': X_train, 'cae_input': att_train},
                                                               )  # batch_size=2048)
        return intermediate_output


class CNN_module():
    '''
    classdocs
    '''
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5
    batch_size = batch_size

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
            # model_internal.add(Conv2D(
            #     nb_filters, i, emb_dim, activation="relu"))
            model_internal.add(Conv2D(nb_filters, (i, emb_dim), activation="relu",
                                      name='conv2d_' + str(i), input_shape=(self.max_len, emb_dim, 1)))
            model_internal.add(BatchNormalization())
            # model_internal.add(MaxPooling2D(
            #     pool_size=(self.max_len - i + 1, 1)))
            model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1), name='maxpool2d_' + str(i)))
            model_internal.add(Flatten())
            flatten = model_internal(reshape)
            flatten_.append(flatten)

        '''Fully Connect Layer & Dropout Layer'''
        # self.model.add_node(Dense(vanila_dimension, activation='tanh'),
        #                     name='fully_connect', inputs=['unit_' + str(i) for i in filter_lengths])
        fully_connect = Dense(vanila_dimension, name='fully_connect')(concatenate(flatten_, axis=-1))
        batch_normalization = BatchNormalization()(fully_connect)
        activation = Activation('tanh')(batch_normalization)
        # self.model.add_node(Dropout(dropout_rate),
        #                     name='dropout', input='fully_connect')
        dropout = Dropout(dropout_rate, name='dropout')(activation)
        '''Projection Layer & Output Layer'''
        # self.model.add_node(Dense(projection_dimension, activation='tanh'),
        #                     name='projection', input='dropout')
        pj = Dense(projection_dimension, activation='tanh', name='output')  # output layer
        projection = pj(dropout)

        # Output Layergit
        model = Model(inputs=doc_input, outputs=projection)
        model.compile(optimizer='rmsprop', loss='mse')
        self.model = model

        # write model summary
        model_summary = open('model_summary', 'w')
        self.model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
        # plot_model(model, to_file='model_cnn.png',show_shapes=True)

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, item_weight, seed, callbacks_list):
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
                                 sample_weight={'output': item_weight}, callbacks=callbacks_list)

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


class CAE_module():
    '''
    classdocs
    '''
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5
    # batch_size = batch_size
    batch_size = 256

    def __init__(self, output_dimesion, cae_N_hidden=50, nb_features=17):
        projection_dimension = output_dimesion

        ''' Attributes module '''
        lam = 1e-3
        # Number of features per data sample
        N = nb_features

        # input layer
        att_input = Input(shape=(N,), name='cae_input')
        # encoded-input layer
        encoded = Dense(cae_N_hidden, activation='tanh', name='encoded')(att_input)
        # decoded-output layer
        att_output = Dense(N, activation='linear', name='cae_output')(encoded)

        # Contractive auto-encoder loss
        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)

            W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x cae_N_hidden
            W = K.transpose(W)  # cae_N_hidden x N
            h = model.get_layer('encoded').output
            dh = h * (1 - h)  # N_batch x cae_N_hidden

            # N_batch x cae_N_hidden * cae_N_hidden x 1 = N_batch x 1
            contractive = lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

            return mse + contractive

        model = Model(inputs=[att_input], outputs=[encoded, att_output])
        model.compile(optimizer='adam',
                      # optimizer={'joint_output': 'rmsprop', 'cae_output':  'adam'},
                      loss={'encoded': 'mse', 'cae_output': contractive_loss},
                      loss_weights={'encoded': 1., 'cae_output': 1.})
        # plot_model(model, to_file='model.png')

        self.model = model
        model_summary = open('model_summary', 'w')
        self.model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
        # plot_model(model, to_file='model_cae.png',show_shapes=True)

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

    def train(self,att_train, V, item_weight, seed,  callbacks_list):
        np.random.seed(seed)
        V = np.random.permutation(V)

        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        np.random.seed(seed)
        att_train = np.random.permutation(att_train)

        print("Train...CAE module")
        history = self.model.fit({'cae_input': att_train},
                                 {'encoded': V, 'cae_output': att_train},
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight={'encoded': item_weight}, callbacks=callbacks_list)
        return history

    def get_projection_layer(self, att_train):
        Y = self.model.predict(
            {'cae_input': att_train}, batch_size=2048)
        return Y[0]


class CNN_CAE_transfer_module():
    '''
    classdocs
    '''
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5
    batch_size = batch_size

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters,
                 init_W=None, cae_N_hidden=50, nb_features=17):

        ''' CNN Module'''
        model_summary = open('model_summary', 'w')
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
            # model_internal.add(Conv2D(
            #     nb_filters, i, emb_dim, activation="relu"))
            model_internal.add(Conv2D(nb_filters, (i, emb_dim), activation="relu",
                                      name='conv2d_' + str(i), input_shape=(self.max_len, emb_dim, 1)))
            # model_internal.add(MaxPooling2D(
            #     pool_size=(self.max_len - i + 1, 1)))
            model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1), name='maxpool2d_' + str(i)))
            model_internal.add(Flatten())
            # plot_model(model_internal, 'model_cnn_cae_transfer_conv2d_%d.png'%i, show_shapes=True)
            model_internal.summary(print_fn=lambda x: model_summary.write(x + '\n'))
            flatten = model_internal(reshape)
            flatten_.append(flatten)

        ''' Attributes module '''
        lam = 1e-3
        N = nb_features
        # cae_N_hidden = 50

        att_input = Input(shape=(N,), name='cae_input')
        encoded = Dense(cae_N_hidden, activation='tanh', name='encoded')(att_input)
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

        '''Transfer layer '''
        # if cae_N_hidden != nb_filters:
        #     sys.exit("For the transfer layer to work ''for now'' the attributes latent vector dimension (--att_dim)"
        #              " must equal the number of filters (kernal) of the conv. layer (--num_kernel_per_ws)")
        # combine the outputs of boths modules
        model_internal = Sequential(name='Transfer_ResBlock')
        model_internal.add(Conv1D(cae_N_hidden / 2, 1, activation="relu",
                                  name='Res_conv2d_1', input_shape=(cae_N_hidden, 1)))
        model_internal.add(Conv1D(cae_N_hidden, 1, activation="relu", name='Res_conv2d_2'))
        model_internal.add(MaxPooling1D(pool_size=cae_N_hidden, name='Res_maxpool1d'))
        model_internal.add(Flatten())

        reshape = Reshape(target_shape=(cae_N_hidden, 1), name='reshape_encoded')(encoded)  # chanels last
        residual = model_internal(reshape)
        transfered = residual

        # plot_model(model_internal,'model_cnn_cae_transfer_transferblock.png',show_shapes=True)
        model_internal.summary(print_fn=lambda x: model_summary.write(x + '\n'))
        # shortcut = encoded
        # transfered = add([residual, shortcut])

        ''' Adding CAE output to CNN output '''
        flatten_.append(transfered)

        '''Fully Connect Layer & Dropout Layer'''
        # self.model.add_node(Dense(vanila_dimension, activation='tanh'),
        #                     name='fully_connect', inputs=['unit_' + str(i) for i in filter_lengths])
        fully_connect = Dense(vanila_dimension, activation='tanh',
                              name='fully_connect')(concatenate(flatten_, axis=-1))

        # self.model.add_node(Dropout(dropout_rate),
        #                     name='dropout', input='fully_connect')
        dropout = Dropout(dropout_rate, name='dropout')(fully_connect)
        # joint_output = add([dropout, transfered], name='joint_output')

        '''Projection Layer & Output Layer'''
        # self.model.add_node(Dense(projection_dimension, activation='tanh'),
        #                     name='projection', input='dropout')
        pj = Dense(projection_dimension, activation='tanh', name='joint_output')  # output layer
        projection = pj(dropout)

        # Output Layergit
        model = Model(inputs=[doc_input, att_input], outputs=[projection, att_output])
        # todo: check the optimizer
        model.compile(optimizer='rmsprop',
                      # optimizer={'joint_output': 'rmsprop', 'cae_output':  'adam'},
                      loss={'joint_output': 'mse', 'cae_output': contractive_loss},
                      loss_weights={'joint_output': 1., 'cae_output': 1.})
        # plot_model(model, to_file='model.png')

        self.model = model
        # plot_model(model, to_file='model_cnn_cae_transfer.png',show_shapes=True)
        model.summary(print_fn=lambda x: model_summary.write(x + '\n'))

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, item_weight, seed, att_train, callbacks_list):
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
                                 sample_weight={'joint_output': item_weight}, callbacks=callbacks_list)

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

    def get_intermediate_output(self, X_train, att_train):
        layer_name = 'concatenated_output'
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer_name).output)
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        intermediate_output = intermediate_layer_model.predict({'doc_input': X_train, 'cae_input': att_train},
                                                               )  # batch_size=2048)
        return intermediate_output


class Stacking_NN_CNN_CAE():
    '''
    classdocs
    '''
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 15
    batch_size = batch_size

    def __init__(self, output_dimesion, input_dim, dropout_rate=0.15,hidden_dim=300):
        ''' CNN Module'''
        model_summary = open('model_summary', 'w')
        vanila_dimension = 200
        projection_dimension = output_dimesion

        # input
        # theta_gamma_input = Input(shape=(input_dim,), dtype='float32', name='theta_gamma')

        model = Sequential()
        model.add(Dense(hidden_dim, activation='relu', input_dim=input_dim))
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_dim, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dimesion, activation='tanh', name='output_layer'))
        model.compile(optimizer='rmsprop', loss='mse')
        # plot_model(model, to_file='model.png')

        self.model = model
        # plot_model(model, to_file='model_cnn_cae_transfer.png',show_shapes=True)
        model.summary(print_fn=lambda x: model_summary.write(x + '\n'))

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, item_weight, seed, callbacks_list):
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)

        np.random.seed(seed)
        V = np.random.permutation(V)

        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...stacking module")
        history = self.model.fit(X_train, V, verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight=item_weight, callbacks=callbacks_list)

        return history

    def get_projection_layer(self, X_train):
        # Y = self.model.predict(
        #     {'doc_input': X_train, 'cae_input':att_train}, batch_size=len(X_train))
        Y = self.model.predict( X_train)
        return Y
