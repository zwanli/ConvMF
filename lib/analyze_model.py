from text_analysis.models import CNN_CAE_transfer_module
from data_manager import Data_Factory
from keras.models import load_model
from keras.utils import plot_model

def main():
    weights_file ='/home/wanliz/data/Extended_ctr/convmf/citeulike_a_extended/results/31-7_in-matrix_no-val_0.01-100-w-cnn-200_cae-100_transfer_no-sc/fold-1/CNN_CAE_weights.hdf5'

    data_factory = Data_Factory()
    model = CNN_CAE_transfer_module(200, 8001, 0.2,
                                    200, 300, 200, None, cae_N_hidden=100,
                                    nb_features=17)
    model.load_model(weights_file)
    plot_model(model.model,to_file='/home/wanliz/model.png',show_layer_names=True,show_shapes=True)
    print ''

if __name__ == '__main__':
     main()
