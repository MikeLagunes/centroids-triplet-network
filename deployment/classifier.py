from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import time
import numpy as np
from tqdm import tqdm
import argparse
import pickle

from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model

import time


def KNN_precision(args):

    npzfile_train = np.load(args.train_file)
    npzfile_test = np.load(args.test_file)


    X_train = npzfile_train['embeddings'][1:]
    y_train = npzfile_train['lebels'][1:]
    filenames_train = npzfile_train['filenames'][1:]

    #print (X_train.shape, y_train.shape)

    X_test = npzfile_test['embeddings'][1:]
    y_test = npzfile_test['lebels'][1:]
    filenames_test = npzfile_test['filenames'][1:]

    #neigh = MLPClassifier(n_neighbors=100,n_jobs=-1)
    start = time.time()
    model = KNeighborsClassifier(n_neighbors=5)#linear_model.LogisticRegression(C=1e5)  # linear_model.LogisticRegression(C=1e5)
    # svm.LinearSVC(C = 1.0) #KNeighborsClassifier(n_neighbors=5) #5 # linear_model.LogisticRegression(C=1e5)

    model.fit(X_train, y_train)
    end = time.time()
    print("Classifier trained in: {:.3f} ".format(end - start))

    # save the model to disk
    filename = 'toybox_knn.sav'
    pickle.dump(model, open(filename, 'wb'))

    total = len(y_test-1)
    correct = 0

    correct += (model.predict(X_test) == y_test).sum()

    print (total)
    print("Precision: {:.3f} ".format(100.*correct/total ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--train_file', nargs='?', type=str, default="/media/alexa/DATA/Miguel/checkpoints_alexa/ae_cvpr/sae_full_same/toybox/train_code_ae_extra_softmax_toybox_best_maxpool_full.npz",
                        help='Path of the input image')
    parser.add_argument('--test_file', nargs='?', type=str, default="/media/alexa/DATA/Miguel/checkpoints_alexa/ae_cvpr/sae_full_same/toybox/train_code_ae_extra_softmax_toybox_best_maxpool_full.npz",
                        help='Path of the input image')
  
    args = parser.parse_args()
    KNN_precision(args)
