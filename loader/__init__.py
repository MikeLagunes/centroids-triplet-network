import json

from loader.cnn_icub import cnn_icub
from loader.cnn_icub_stability import cnn_icub_stability
from loader.cnn_icub_warp import cnn_icub_warp
from loader.double_resnet_icub_softmax import double_resnet_icub_softmax
from loader.triplet_resnet_icub_softmax import triplet_resnet_icub_softmax
from loader.triplet_resnet_icub_temporal import triplet_resnet_icub_temporal

from loader.cnn_household import cnn_household
#from loader.cnn_household_warp import cnn_household_warp
#from loader.cnn_household_stability import cnn_household_stability
from loader.temporal_triplet_resnet_household import temporal_triplet_resnet_household
from loader.triplet_resnet_household_softmax import triplet_resnet_household_softmax
from loader.double_resnet_household_softmax import double_resnet_household_softmax
from loader.triplet_resnet_household_temporal import triplet_resnet_household_temporal

from loader.cnn_toybox import cnn_toybox
from loader.cnn_toybox_warp import cnn_toybox_warp
#from loader.cnn_toybox_stability import cnn_toybox_stability
from loader.double_resnet_toybox_softmax import double_resnet_toybox_softmax
from loader.triplet_resnet_toybox_softmax import triplet_resnet_toybox_softmax
from loader.triplet_resnet_toybox_temporal import triplet_resnet_toybox_temporal


from loader.cnn_core50 import cnn_core50
#from loader.cnn_core50_warp import cnn_core50_warp
#from loader.cnn_core50_stability import cnn_core50_stability
from loader.triplet_resnet_core50_softmax import triplet_resnet_core50_softmax
from loader.double_resnet_core50_softmax import double_resnet_core50_softmax
from loader.triplet_resnet_core50_temporal import triplet_resnet_core50_temporal
from loader.double_triplet_resnet_core50 import double_triplet_resnet_core50
from loader.triplet_resnet_core50_neighboring import triplet_resnet_core50_neighboring


def get_loader(name):
    """get_loader

    :param name:
    """
    return {

        'cnn_icub':cnn_icub,
        'cnn_icub_warp':cnn_icub_warp,
        'cnn_icub_stability':cnn_icub_stability,
        'double_resnet_icub_softmax':double_resnet_icub_softmax,
        'triplet_resnet_icub_softmax':triplet_resnet_icub_softmax,
        'triplet_resnet_icub_temporal':triplet_resnet_icub_temporal,

        'cnn_household':cnn_household,
#        'cnn_household_warp':cnn_household_warp,
#        'cnn_household_stability':cnn_household_stability, 
        'temporal_triplet_resnet_household':temporal_triplet_resnet_household,
        'triplet_resnet_household_temporal':triplet_resnet_household_temporal,
        'triplet_resnet_household_softmax':triplet_resnet_household_softmax,
        'double_resnet_household_softmax':double_resnet_household_softmax,
        
        'cnn_toybox':cnn_toybox,
        'cnn_toybox_warp':cnn_toybox_warp,
#        'cnn_toybox_stability':cnn_toybox_stability,
        'double_resnet_toybox_softmax': double_resnet_toybox_softmax,
        'triplet_resnet_toybox_softmax': triplet_resnet_toybox_softmax,
        'triplet_resnet_toybox_temporal': triplet_resnet_toybox_temporal,
         
        'cnn_core50':cnn_core50,
#        'cnn_core50_warp':cnn_core50_warp,
#        'cnn_core50_stability':cnn_core50_stability,
        'triplet_resnet_core50_softmax': triplet_resnet_core50_softmax,
        'double_resnet_core50_softmax': double_resnet_core50_softmax,
        'triplet_resnet_core50_temporal':triplet_resnet_core50_temporal,
        #'double_resnet_core50_softmax':double_triplet_resnet_core50,
        'triplet_resnet_core50_neighboring':triplet_resnet_core50_neighboring,
    }[name]


#def get_data_path(name, config_file='../config.json'):
def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
