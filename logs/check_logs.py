from time import time
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import matplotlib
import matplotlib.pyplot as plt



def main(args):

    logs_file = args.logs_path
    accuracies_file  = args.accuracies_path

    npzfile_logs = np.load(logs_file)
    npzfile_accs = np.load(accuracies_file)

    # Logs

    # log_ID = npzfile_logs['ID']
    log_steps = npzfile_logs['steps']#[8:]
    # log_losses_softmax = npzfile_logs['losses_softmax']
    # log_losses_triplet = npzfile_logs['losses_triplet']
    # log_losses_rec = npzfile_logs['losses_rec']
    # log_losses_sum = npzfile_logs['losses_sum']
    
    #log_losses_nby = npzfile_logs['losses_nby']

    #print(log_steps)

    # Accuracies

    acc_steps = npzfile_accs['steps']
    acc_all = npzfile_accs['accuracies_all']

    #print(log_ID)
    #print(acc_all)
   
 #.pkl\n81.8540804814'


    # acc_all = [float(x[x.find("\n")+1:]) for x in acc_all]
    # acc_novel = [float(x[x.find("\n")+1:]) for x in acc_novel]
    # acc_known = [float(x[x.find("\n")+1:]) for x in acc_known]

    print(acc_all)
    # print(acc_steps)
    
    idx = np.where(acc_all==max(acc_all))# acc_all.find(max(acc_all))
  
    print("Accuracy full: {} ".format(max(acc_all)))
    


    plt.subplot(1, 2, 1)
    plt.plot(log_steps, losses_softmax)
    #plt.plot(log_steps, log_losses_softmax)
    plt.xlim(log_steps[0], log_steps[-1])
    plt.xlabel('Steps')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.title('Loss value')

    plt.subplot(1, 2, 2)
    plt.plot(acc_steps, acc_all, label="Full")
    plt.xlim(log_steps[0], log_steps[-1])
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
 

    plt.subplot(1, 2, 2)
    plt.plot(acc_steps, acc_known, label="Known")
    plt.xlim(log_steps[0], log_steps[-1])
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
  

    plt.subplot(1, 2, 2)
    plt.plot(acc_steps, acc_novel, label="Novel")
    plt.xlim(log_steps[0], log_steps[-1])
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.title('Accuracy ')

    plt.tight_layout()
    leg = plt.legend()


    plt.show()

    return

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--logs_path', nargs='?', type=str, default="/home/mikelf/Downloads/logs/core50_triplet_exemplar_softmax_train_log_0_1e-2.npz",
                        help='Path of the input image')
    parser.add_argument('--accuracies_path', nargs='?', type=str, default="/home/mikelf/Downloads/logs/core50_triplet_exemplar_softmax_accuracies_log_0_1e-2.npz",
                        help='Path of the input image')

#/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints_alexa/AE/TAES/v2/hinter_l02/hinterstoisser_triplet_ae_softmax_accuracies_log_test_1_lambda_02.npz
#home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints_alexa/AE/TAES/v2/hinter_l02/hinterstoisser_triplet_ae_softmax_train_log_test_1_lambda_02.npz
    args = parser.parse_args()
    main(args)


    #/home/mikelf/Downloads/logs/core50_triplet_exemplar_softmax_accuracies_log_0_0_x1.npz
