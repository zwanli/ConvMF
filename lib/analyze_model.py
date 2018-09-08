from keras.utils import plot_model

from data_manager import Data_Factory
from text_analysis.models import CNN_CAE_transfer_module, CNN_module, CNN_CAE_module
import matplotlib.pyplot as plt
import numpy as np
import os
def plot_weights(weights,exp_dir):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    print(weights.shape)
    prev_y = np.arange(0, weights.shape[0], 1)+1
    prev_y *= 500
    prev_x= [5]*weights.shape[0]
    curr_y = np.arange(0, weights.shape[1], 1)* 16
    curr_x = [15]*weights.shape[1]

    colors={0:'b',1:'g',2:'r',3:'c',4:'m',5:'y'}

    normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    plt.ion()
    plt.figure(figsize=(16, 12), dpi=300)
    plt.subplot(211)

    print (plt.rcParams['interactive'])

    plt.scatter(prev_x, prev_y, s=50)
    plt.scatter(curr_x, curr_y, s=1, c='g')
    plt.title('Fully connected layer weights')
    for i in range(weights.shape[0]):
        plt.annotate('CAE' if i == weights.shape[0]-1 else 'CNN', (prev_x[i], prev_y[i]))
        for j in range (weights.shape[1]):
            plt.plot([prev_x[i], curr_x[j]], [prev_y[i], curr_y[j]],colors[i%len(colors)],alpha=0.5, lw=normalized[i, j]/5)
    plt.savefig(os.path.join(exp_dir,'FC-layer-weights.png'),dpi=300)
    plt.show()

def main():
    #cnn_cae_transfer
    # exp_dir = '/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/5-8_in-matrix_no-val_0.01-100_w_cnn-100_cae-50_transfer-2_no-sc//fold-4/'
    exp_dir = '/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/6-9_in-matrix-200_no-val_0.01-100-w-cnn-100_cae-25_transfer-noSC/fold-4/'
    exp_dir = '/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/7-9_in-matrix-200_no-val_0.01-100-w-cnn-50_cae-150_transfer-noSC/fold-1'

    #cnn_cae_concat
    # exp_dir ='/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/6-8_out-of-matrix_no-val_1-1000-w-cnn-100_cae-50_concat/fold-1/'
    #cnn
    # exp_dir = '/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/6-8_in-matrix_no-val_0.01-100_w_cnn/fold-1'
    fc_weights_file =os.path.join(exp_dir,'FC_weights.npy')
    module = 'cnn_cae_transfer'
    ''' Network parameters'''
    nb_filters = 50
    cae_N_hidden = 150
    nb_features = 23
    if module == 'cnn_cae_transfer' or module == 'cnn_cae_concat':
        weights_file = os.path.join(exp_dir, 'CNN_CAE_weights.hdf5')
    elif module == 'cnn':
        weights_file = os.path.join(exp_dir, 'CNN_weights.hdf5')
    if not os.path.exists(fc_weights_file):
        if module == 'cnn_cae_transfer':
            model = CNN_CAE_transfer_module(output_dimesion= 200, vocab_size=8001, dropout_rate=0.2,
                                        emb_dim=200, max_len=300, nb_filters=nb_filters, init_W=None, cae_N_hidden=cae_N_hidden,
                                           nb_features=nb_features)
        elif module =='cnn_cae_concat':
            model =CNN_CAE_module(output_dimesion= 200, vocab_size=8001, dropout_rate=0.2,
                                        emb_dim=200, max_len=300, nb_filters=nb_filters, init_W=None, cae_N_hidden=cae_N_hidden,
                                           nb_features=17)
        elif module == 'cnn':
            model = CNN_module(output_dimesion=200, vocab_size=8001, dropout_rate=0.2,
                                            emb_dim=200, max_len=300, nb_filters=nb_filters, init_W=None)
        model.load_model(weights_file)
        if module == 'cnn_cae_concat':
            layer_name = 'joint_output'
        else:
            layer_name = 'fully_connect'
        weights = model.model.get_layer(layer_name).get_weights()
        bias = weights[1]
        fc_weights = weights[0]
        np.save(fc_weights_file,fc_weights)
    else:
        fc_weights = np.load(fc_weights_file)
    # Normalized Data
    normalized = (fc_weights - np.min(fc_weights)) / (np.max(fc_weights) - np.min(fc_weights))
    cnn_output ={}


    # nb_filters = cae_N_hidden
    if module == 'cnn_cae_concat':
        cnn_vector_length = 2
    else:
        cnn_vector_length = 3#normalized.shape[0] / nb_filters
    for i in range(cnn_vector_length):
        print (i*nb_filters,i*nb_filters+(nb_filters-1))
        cnn_output[i] = np.sum(normalized[i*nb_filters:i*nb_filters+nb_filters-1],axis=0,keepdims=True)
    agg_output = np.vstack(cnn_output.values())
    if fc_weights.shape[0] > cnn_vector_length * nb_filters:
        #CAE output
        cae_output = np.sum(normalized[cnn_vector_length * nb_filters:],axis=0,keepdims=True)
        agg_output = np.vstack((agg_output,cae_output))
    plot_weights(agg_output,exp_dir)

    plot_model(model.model,to_file='/home/wanliz/model.png',show_layer_names=True,show_shapes=True)
    print ('')


if __name__ == '__main__':
     main()
