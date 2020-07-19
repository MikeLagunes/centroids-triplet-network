import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import random

import sys, os
sys.path.append('.')
sys.path.append('..')

from torch.autograd import Variable
from torch.utils import data
from torch import optim

import subprocess
import yaml

losses_softmax = []
losses_triplet = []
losses_rec = []
losses_sum = []
losses_nby = []
accuracies_all = []
accuracies_known = []
accuracies_novel = []

num_class = 0


losses = [losses_softmax, losses_triplet, losses_rec, losses_sum, losses_nby]
losses_id = ["loss_softmax", "loss_triplet", "loss_rec", "loss_sum", "loss_cetroids"]

accuracies = [accuracies_all, accuracies_known, accuracies_novel]
accuracies_id = ["accuracies_all", "accuracies_known", "accuracies_novel"]

steps = []
accuracy_step = []

ID = ""

args_local = None
ckpt_full_path =  ""


def save_checkpoint(epoch, model, optimizer, description):

    global ckpt_full_path

    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'optimizer_state' : optimizer.state_dict(),}

    ckpt_full_path = "{}/{}_{}_{}_{}.pkl".format( args_local.ckpt_path, args_local.arch, args_local.dataset, description, args_local.id)

    torch.save(state, "{}/{}_{}_{}_{}.pkl".format( args_local.ckpt_path, args_local.arch, args_local.dataset, description, args_local.id))
    return


def show_setup(args, n_classes, optimizer, loss_fn):

    global args_local, ID, num_class
    args_local=args

    print("Model: {} | Training on: {} | Number of Classes: {}".format(args.arch, args.dataset, n_classes))
    print("Embedding size: {}".format(args.embedding_size))
    print("Epochs: {}".format(args.n_epoch))
    print("Optimizer: {}".format(optimizer))
    print("Loss function: {}".format(loss_fn))


    ID = "Model: {} \n Training on: {} \n Number of Classes: {} \n Epochs: {} \n Optimizer: {} \n Loss function: {} ".format(args.arch, 
        args.dataset, n_classes,args.n_epoch, optimizer, loss_fn)

    num_class = n_classes

    return 


def get_instances(args):

    #global args_local
    #args_local=args

    gt_file = 'loader/dataset_splits.yml'

    #gt_file = '../loader/dataset_splits.yml' #../ for local

    with open(gt_file, 'r') as f:
        doc = yaml.load(f, Loader=yaml.FullLoader) #

    instances = doc[args.dataset][args.instances]

    return instances


def log_loss(epoch, step, loss_softmax=None, loss_triplet=None, loss_rec=None, loss_sum=None, loss_nby=None):


    global steps, losses_softmax, losses_triplet, losses_rec, losses_sum, losses_nby, losses, args_local, ID


    if loss_softmax != None: losses_softmax.append(loss_softmax)
    if loss_triplet != None: losses_triplet.append(loss_triplet)
    if loss_rec != None: losses_rec.append(loss_rec)
    if loss_sum != None: losses_sum.append(loss_sum)
    if loss_nby != None: losses_nby.append(loss_nby)

   # print(loss_nby)

    steps.append(step)

    log_message = "" 

    for idx in range(len(losses)):

        if len(losses[idx]) > 0:
    
            log_message += losses_id[idx] + ": %.4f | " % losses[idx][-1] 

    print("Epoch [{}] | {} ".format(epoch, log_message))

    if step % args_local.logs_freq == 0:
        np.savez("{}/{}_{}_train_log_{}".format(args_local.ckpt_path, args_local.dataset, args_local.arch, args_local.id),  steps=steps, losses_softmax=losses_softmax, 
            losses_triplet=losses_triplet, losses_rec=losses_rec, losses_sum=losses_sum, losses_nby=losses_nby, ID=ID)

    return

def eval_model_real(step, sets):

    global ckpt_full_path
    global args_local, num_class
    global accuracies_all, accuracies_known, accuracies_novel, accuracies, ID, accuracy_step

    sets = sets

    sets_to_test = ["known"]
 

    for set_to_test in sets_to_test:

        #print(args_local.embedding_size)

        subprocess.call(["python3 test/test_embeddings.py --ckpt_path={} --dataset={} --instances={} --split=test --embedding_size={}".format(ckpt_full_path,  args_local.dataset, set_to_test, args_local.embedding_size)], shell=True)
        subprocess.call(["python3 test/test_embeddings.py --ckpt_path={} --dataset={} --instances={} --split=train --embedding_size={}".format(ckpt_full_path,  args_local.dataset, set_to_test, args_local.embedding_size)], shell=True)
        

      
        subprocess.call(["python3 test/nearest_neighbours_exemplars.py --ckpt_path={} --instances={} --num_classes={} --embedding_size={}".format(ckpt_full_path, set_to_test, num_class, args_local.embedding_size )], shell=True)


    return 0

def eval_model(step, sets):

    global ckpt_full_path
    global args_local, num_class
    global accuracies_all, accuracies_known, accuracies_novel, accuracies, ID, accuracy_step

    sets = sets

    if sets == "all": sets_to_test = ["full", "known", "novel"]
    elif sets == "known": sets_to_test = ["known"]
    elif sets == "novel": sets_to_test = ["novel"]
    elif sets == "full": sets_to_test = ["full"]

    for set_to_test in sets_to_test:

        #print(args_local.embedding_size)

        subprocess.call(["python3 test/test_embeddings.py --ckpt_path={} --dataset={} --instances={} --split=test --embedding_size={}".format(ckpt_full_path,  args_local.dataset, set_to_test, args_local.embedding_size)], shell=True)
        subprocess.call(["python3 test/test_embeddings.py --ckpt_path={} --dataset={} --instances={} --split=train --embedding_size={}".format(ckpt_full_path,  args_local.dataset, set_to_test, args_local.embedding_size)], shell=True)
        
        #print (ckpt_full_path)
        if set_to_test == "full":
            accuracy_all = subprocess.check_output(["python3 test/nearest_neighbours_exemplars.py --ckpt_path={} --instances={} --num_classes={} --embedding_size={}".format(ckpt_full_path, set_to_test, num_class, args_local.embedding_size)], shell=True)
            accuracy_all = float(accuracy_all.decode('utf-8'))
            accuracies_all.append(accuracy_all)
            print ("Accuracy full: {:.3f} | step: {} ".format(accuracy_all, step))

        elif set_to_test == "known": 
            accuracy_known = subprocess.check_output(["python3 test/nearest_neighbours_exemplars.py --ckpt_path={} --instances={} --num_classes={} --embedding_size={}".format(ckpt_full_path, set_to_test, num_class, args_local.embedding_size )], shell=True)
            accuracy_known = float(accuracy_known.decode('utf-8'))
            accuracies_known.append(accuracy_known)
            #accuracy_all = accuracy_known

            print ("Accuracy known: {:.3f} | step: {} ".format(accuracy_known, step)) 
             
        elif set_to_test == "novel":
            accuracy_novel = subprocess.check_output(["python3 test/nearest_neighbours_exemplars.py --ckpt_path={} --instances={} --num_classes={} --embedding_size={}".format(ckpt_full_path, set_to_test, num_class, args_local.embedding_size )], shell=True)
            accuracy_novel = float(accuracy_novel.decode('utf-8'))
            accuracies_novel.append(accuracy_novel) 
            print ("Accuracy novel: {:.3f} | step: {} ".format(accuracy_novel, step))


    accuracy_step.append(step)
    #print ("OK")

    np.savez("{}/{}_{}_accuracies_log_{}".format(args_local.ckpt_path, args_local.dataset, args_local.arch, args_local.id),  steps=accuracy_step, accuracies_all=accuracies_all, 
            accuracies_known=accuracies_known, accuracies_novel=accuracies_novel, ID=ID)


    return accuracy_all
