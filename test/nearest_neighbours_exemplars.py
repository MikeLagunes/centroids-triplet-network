from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from time import time
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import matplotlib.pyplot as plt
from numpy import linalg as LA

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def KNN_precision():


    train_file = os.path.split(args.ckpt_path)[0]+"/train_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"
    test_file  = os.path.split(args.ckpt_path)[0]+"/test_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"


    npzfile_train = np.load(train_file)
    npzfile_test = np.load(test_file)


    X_train = npzfile_train['embeddings'][1:]
    y_train = npzfile_train['lebels'][1:]
    filenames_train = npzfile_train['filenames'][1:]

    #exemplars = np.zeros((args.num_classes,args.embedding_size))
    #exemplars_labels = np.zeros((args.num_classes,1))
    #exemplars_counter = np.zeros((args.num_classes,1))
    #exemplars_counter = np.zeros((50,1))
    #exemplars = np.asarray(exemplars)

    #print("embed python: ", X_train[0], filenames_train[0])


    #for i in range(len(y_train)):
        #print(exemplars[int(y_train[i])].shape, X_train[i].shape)
    #    exemplars[int(y_train[i])] = exemplars[int(y_train[i])] + X_train[i]
    #    exemplars_counter[int(y_train[i])] += 1
    #    exemplars_labels[int(y_train[i])] = y_train[i]

    #exemplars_labels = np.ravel(exemplars_labels)


    #for i in range (len(exemplars_labels)):
        #exemplars[i] = exemplars[i] / (exemplars_counter[i] - 1)
    #    norm = LA.norm(exemplars[i])
    #    if norm != 0:
    #        exemplars[i] = exemplars[i]/norm



    #print("In python: ",exemplars[0])

    X_test = npzfile_test['embeddings'][1:] #1000
    y_test = npzfile_test['lebels'][1:]
    #filenames_test = npzfile_test['filenames']#[1:1000]
    

    #Temporal integration

    # for i in range(4,len(y_test)):
    #     X_test[i] = (X_test[i]+X_test[i-1]+X_test[i-2]+X_test[i-3]+X_test[i-4])/5  

    clf = NearestCentroid()
    clf.fit(X_train, y_train)

    # neigh = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, p=2)
    # neigh.fit(exemplars, exemplars_labels)

    

    #total = len(y_test-1)
    #correct = 0
    #correct += (clf.predict(X_test) == y_test).sum()

    correct = clf.score(X_test, y_test)

    sys.stdout.write("{}".format(100.*correct))
    sys.stdout.flush()
    sys.exit(0)
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--ckpt_path', nargs='?', type=str, default="",
                        help='Path of the input image')
    parser.add_argument('--instances', nargs='?', type=str, default="full", 
                        help='Path of the input image')
    parser.add_argument('--num_classes', nargs='?', type=int, default=10,
                        help='Number of classes in dataset')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128,
                        help='Number of classes in dataset')

    args = parser.parse_args()
    KNN_precision()