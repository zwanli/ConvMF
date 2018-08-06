from keras.utils import plot_model

from data_manager import Data_Factory
from text_analysis.models import CNN_CAE_transfer_module, CNN_module
import matplotlib.pyplot as plt
import numpy as np
import os
def draw_weights(weights,exp_dir):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    print(weights.shape)
    prev_y = np.arange(0, weights.shape[0], 1)+1
    prev_y *= 500
    prev_x= [5]*weights.shape[0]
    curr_y = np.arange(0, weights.shape[1], 1)* 16
    curr_x = [15]*weights.shape[1]

    normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    plt.ion()
    plt.figure(figsize=(16, 12), dpi=300)
    plt.subplot(211)

    print (plt.rcParams['interactive'])

    plt.scatter(prev_x, prev_y, s=50)
    plt.scatter(curr_x, curr_y, s=1, c='g')
    plt.title('Fully connected layer weights')
    for i in range(weights.shape[0]):
        plt.annotate('CAE' if i ==3 else 'CNN', (prev_x[i], prev_y[i]))
        for j in range (weights.shape[1]):
            plt.plot([prev_x[i], curr_x[j]], [prev_y[i], curr_y[j]], lw=normalized[i, j])
    plt.savefig(os.path.join(exp_dir,'FC-layer-weights.png'),dpi=300)
    plt.show()

def main():
    #cnn_cae_transfer
    exp_dir = '/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/5-8_in-matrix_no-val_0.01-100_w_cnn-100_cae-50_transfer-2_no-sc//fold-4/'
    #cnn
    # exp_dir = '/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/6-8_in-matrix_no-val_0.01-100_w_cnn/fold-1'
    fc_weights_file =os.path.join(exp_dir,'FC_weights.npy')
    module = 'cnn_cae_transfer'
    if module == 'cnn_cae_transfer':
        weights_file = os.path.join(exp_dir, 'CNN_CAE_weights.hdf5')
    elif module == 'cnn':
        weights_file = os.path.join(exp_dir, 'CNN_weights.hdf5')
    if not os.path.exists(fc_weights_file):
        if module == 'cnn_cae_transfer':
            model = CNN_CAE_transfer_module(output_dimesion= 200, vocab_size=8001, dropout_rate=0.2,
                                        emb_dim=200, max_len=300, nb_filters=100, init_W=None, cae_N_hidden=50,
                                           nb_features=17)
        elif module == 'cnn':
            model = CNN_module(output_dimesion=200, vocab_size=8001, dropout_rate=0.2,
                                            emb_dim=200, max_len=300, nb_filters=100, init_W=None)
        model.load_model(weights_file)
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
    for i in range(3):
        print (i*100,i*100+99)
        cnn_output[i] = np.sum(normalized[i*100:i*100+99],axis=0,keepdims=True)
    agg_output = np.vstack(cnn_output.values())
    if fc_weights.shape[0] == 350:
        #CAE output
        cae_output = np.sum(normalized[300:],axis=0,keepdims=True)
        agg_output = np.vstack((agg_output,cae_output))
    draw_weights(agg_output,exp_dir)

    plot_model(model.model,to_file='/home/wanliz/model.png',show_layer_names=True,show_shapes=True)
    print ('')


if __name__ == '__main__':
     main()
