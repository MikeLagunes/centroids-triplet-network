from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from time import time
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import matplotlib.pyplot as plt
from numpy import linalg as LA



def KNN_precision():


    train_file = os.path.split(args.ckpt_path)[0]+"/train_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"
    test_file  = os.path.split(args.ckpt_path)[0]+"/test_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"

    npzfile_train = np.load(train_file)
    npzfile_test = np.load(test_file)


    X_train = npzfile_train['embeddings'][1:]
    y_train = npzfile_train['lebels'][1:]
    filenames_train = npzfile_train['filenames'][1:]

   
    X_test = npzfile_test['embeddings'][1:] 
    y_test = npzfile_test['lebels'][1:]

    clf = NearestCentroid()
    clf.fit(X_train, y_train)

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