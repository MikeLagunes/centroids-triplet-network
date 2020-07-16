import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

import sys, os
sys.path.append('.')

from models import get_model
from loader import get_loader, get_data_path
from utils import convert_state_dict

import csv


def test(args):



    root_dir = args.test_path #"/media/alexa/DATA/Miguel/results/" + args.dataset +"/triplet_cnn/" 

    data_loader = get_loader("cnn_" + args.dataset)

    data_path = get_data_path(args.dataset)
    
    t_loader = data_loader(data_path, is_transform=True, split=args.split, img_size=(args.img_rows, args.img_cols), augmentations=None, class_id='')

    n_classes = t_loader.n_classes

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=0, shuffle=False)
 
    # Setup Model
    # Setup Model
    model = get_model(args.arch, n_classes)
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    weights = torch.load(args.model_path)['model_state']

    #state = convert_state_dict(weights)
    model.load_state_dict(weights)
    model.eval()

    #model.load_state_dict(weights)
    print ("Model Loaded, Epoch: ", torch.load(args.model_path)['epoch'])

    print ("Projecting: " + args.dataset + " | " + args.split + " set" )

    #print(model)
    
    model = model.cuda()
    model.eval()
    #images = Variable(img.cuda(0), volatile=True)

    output_embedding = np.array([])
    outputs_embedding = np.zeros((1,128))
    labels_embedding = np.zeros((1))
    path_imgs = []

    for i, (images, labels, path_img) in enumerate(tqdm(trainloader)):
            
        images = Variable(images.cuda())
        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()
        
        #labels = Variable(labels.cuda())
        outputs, _, _ = model(images, images, images)

        output_embedding = outputs.data
        output_embedding = output_embedding.cpu().numpy()

        outputs_embedding = np.concatenate((outputs_embedding,output_embedding), axis=0)
        labels_embedding = np.concatenate((labels_embedding,labels), axis=0)

        path_imgs.extend(path_img)

      
    np.savez(root_dir + args.split + "_set_triplet_cnn_" + args.dataset,  embeddings=outputs_embedding, lebels=labels_embedding, filenames=path_imgs)

    print ('Done: ')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch', nargs='?', type=str, default='triplet_cnn', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='', 
                        help='Path to the saved model')
    parser.add_argument('--test_path', nargs='?', type=str, default='', 
                        help='Path to saving results')
    parser.add_argument('--dataset', nargs='?', type=str, default='arc', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=5, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='train-item', 
                        help='Dataset split to use [\'train, eval\']')

    args = parser.parse_args()
    test(args)
