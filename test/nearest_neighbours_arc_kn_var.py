from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import time
import numpy as np
from tqdm import tqdm


"""
=============================================================================
Manifold learning on handwritten digits: Locally Linear Embedding, Isomap...
=============================================================================

An illustration of various embeddings on the digits dataset.

The RandomTreesEmbedding, from the :mod:`sklearn.ensemble` module, is not
technically a manifold embedding method, as it learn a high-dimensional
representation on which we apply a dimensionality reduction method.
However, it is often useful to cast a dataset into a representation in
which the classes are linearly-separable.

t-SNE will be initialized with the embedding that is generated by PCA in
this example, which is not the default setting. It ensures global stability
of the embedding, i.e., the embedding does not depend on random
initialization.
"""

# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011

from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

from sklearn.decomposition import IncrementalPCA

from PIL import Image
import csv
import os
# Labels dictionary

known_classes_id = [0,1,2,3,4,5,8,9,11,15,18,19,20,21,22,24,25,26,27,
                 29,30,33,34,35,36,37,40,41,43,45,46,47,49,50,52,53,54,55,56,58,59]

known_classes = {"avery_binder","balloons","band_aid_tape","bath_sponge","black_fashion_gloves","burts_bees_baby_wipes","colgate_toothbrush_4pk","composition_book","crayons","duct_tape","empty","epsom_salts","expo_eraser","fiskars_scissors","flashlight","glue_sticks","hand_weight","hanes_socks","hinged_ruled_index_cards","ice_cube_tray","irish_spring_soap","laugh_out_loud_jokes","marbles","measuring_spoons","mesh_cup","mouse_traps","pie_plates","plastic_wine_glass","poland_spring_water","reynolds_wrap","robots_dvd","robots_everywhere","scotch_sponges","speed_stick","table_cloth","tennis_ball_container","ticonderoga_pencils","tissue_box","toilet_brush","white_facecloth","windex"
}

labels_dic = {
'avery_binder':0,'balloons':1,'band_aid_tape':2,'bath_sponge':3,'black_fashion_gloves':4,'burts_bees_baby_wipes':5,
'cherokee_easy_tee_shirt':6,'cloud_b_plush_bear':7,'colgate_toothbrush_4pk':8,'composition_book':9,'cool_shot_glue_sticks':10,
'crayons':11,'creativity_chenille_stems':12,'dove_beauty_bar':13,'dr_browns_bottle_brush':14,'duct_tape':15,
'easter_turtle_sippy_cup':16,'elmers_washable_no_run_school_glue':17,'empty':18,'epsom_salts':19,'expo_eraser':20,
'fiskars_scissors':21,'flashlight':22,'folgers_classic_roast_coffee':23,'glue_sticks':24,'hand_weight':25,
'hanes_socks':26,'hinged_ruled_index_cards':27,'i_am_a_bunny_book':28,'ice_cube_tray':29,'irish_spring_soap':30,
'jane_eyre_dvd':31,'kyjen_squeakin_eggs_plush_puppies':32,'laugh_out_loud_jokes':33,'marbles':34,'measuring_spoons':35,
'mesh_cup':36,'mouse_traps':37,'oral_b_toothbrush_red':38,'peva_shower_curtain_liner':39,'pie_plates':40,
'plastic_wine_glass':41,'platinum_pets_dog_bowl':42,'poland_spring_water':43,'rawlings_baseball':44,'reynolds_wrap':45,
'robots_dvd':46,'robots_everywhere':47,'scotch_bubble_mailer':48,'scotch_sponges':49,'speed_stick':50,
'staples_index_cards':51,'table_cloth':52,'tennis_ball_container':53,'ticonderoga_pencils':54,'tissue_box':55,
'toilet_brush':56,'up_glucose_bottle':57,'white_facecloth':58,'windex':59,'woods_extension_cord':60}

#=======================================================================================================================

npzfile_test = np.load("/media/mikelf/media_rob/experiments/arc/tae/allvsall2/1/test/test_set_triplet_ae_arc.npz")

#=======================================================================================================================

filename = '/media/mikelf/media_rob/Datasets/arc-novel/ml/test-other-objects-list.txt'


# /media/mikelf/media_rob/experiments/arc/tae/allvsall2/1/test-item/test-item_set_triplet_cnn_avery_binder-item_arc.npz
root_eval = "/media/mikelf/media_rob/experiments/arc/tae/allvsall2/1/test-item/test-item_set_triplet_cnn_"

with open(filename) as f:
    data = f.readlines()

reader = csv.reader(data)

testing_img_idx = 0
label_count = 1


correct = 0

for row in reader:
    row = map(int, row)

    X_test = npzfile_test['embeddings'][testing_img_idx+1]
    y_test = npzfile_test['lebels'][testing_img_idx+1]
    filenames_test = npzfile_test['filenames'][testing_img_idx]

    object_class = os.path.split(os.path.split(filenames_test)[0])[1]



    iter = 0

    #print(len(known_classes_id))

    if row[0]-1 not in known_classes_id:
        #testing_img_idx += 1
        continue

    # if object_class not  in known_classes:
    #     print (object_class, y_test, row)
    #     testing_img_idx += 1
    #     #if testing_img_idx == 562: testing_img_idx -= 1
    #     if testing_img_idx == 389: testing_img_idx -= 1
    #     continue

    label_count += 1



    for label_img in row:
        for key, value in labels_dic.iteritems():

            if key in known_classes:

                if value + 1 == label_img:
                    npzfile_train = np.load(root_eval + key +"-item_arc.npz")

                    X_temp = npzfile_train['embeddings'][1:]
                    y_temp = npzfile_train['lebels'][1:]

                    filenames_temp = npzfile_train['filenames'][1:]

                    if iter == 0:
                        X_train = X_temp
                        y_train = y_temp
                        filenames_train = filenames_temp

                    else:

                        X_train = np.append(X_train, X_temp[:], axis=0)
                        y_train = np.concatenate((y_train, y_temp), axis=0)
                        filenames_train = np.concatenate((filenames_train, filenames_temp), axis=0)

                    iter += 1


    #print (X_train.shape)




    #print(y_test, y_train)
    #print(X_test, X_train)

    #neigh = MLPClassifier(n_neighbors=100,n_jobs=-1)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)

    #print("kNN done - 0.5 margin")

    #prediction = neigh.predict([X_test])

    total = len(npzfile_test['lebels'])

    #print ("Prediction: ", prediction, " | ", "Correct: ", correct)

    correct += neigh.predict([X_test]) == y_test
    testing_img_idx += 1

    #if testing_img_idx == 562: testing_img_idx -= 1
    if testing_img_idx == 389: testing_img_idx -= 1

print (label_count)
print( 1.*correct/(label_count-1) )
