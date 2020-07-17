import json

from loader.cnn_icub import cnn_icub
from loader.triplet_resnet_icub_softmax import triplet_resnet_icub_softmax

from loader.cnn_household import cnn_household
from loader.triplet_resnet_household_softmax import triplet_resnet_household_softmax

from loader.cnn_toybox import cnn_toybox
from loader.triplet_resnet_toybox_softmax import triplet_resnet_toybox_softmax

from loader.cnn_core50 import cnn_core50
from loader.triplet_resnet_core50_softmax import triplet_resnet_core50_softmax


def get_loader(name):
    """get_loader

    :param name:
    """
    return {

        'cnn_icub':cnn_icub,
        'triplet_resnet_icub_softmax':triplet_resnet_icub_softmax,

        'cnn_household':cnn_household,
        'triplet_resnet_household_softmax':triplet_resnet_household_softmax,
        
        'cnn_toybox':cnn_toybox,
        'triplet_resnet_toybox_softmax': triplet_resnet_toybox_softmax,
         
        'cnn_core50':cnn_core50,
        'triplet_resnet_core50_softmax': triplet_resnet_core50_softmax,

    }[name]


#def get_data_path(name, config_file='../config.json'):
def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
