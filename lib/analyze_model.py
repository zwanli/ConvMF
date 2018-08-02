from keras.utils import plot_model

from data_manager import Data_Factory
from text_analysis.models import CNN_CAE_transfer_module


def main():
    weights_file ='/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/2-8_out-of-matrix_no-val_1-1000-w-cnn-200_cae-100_transfer_no-sc/fold-1/CNN_CAE_weights.hdf5'

    data_factory = Data_Factory()
    model = CNN_CAE_transfer_module(output_dimesion= 200, vocab_size=8001, dropout_rate=0.2,
                                    emb_dim=200, max_len=300, nb_filters=200, init_W=None, cae_N_hidden=100,
                                    nb_features=17)

    model.load_model(weights_file)
    plot_model(model.model,to_file='/home/wanliz/model.png',show_layer_names=True,show_shapes=True)
    print ''

if __name__ == '__main__':
     main()
