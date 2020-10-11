from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class SoftmaxLoss(nn.Module):
    """
    Softmax loss
    Takes logits and class labels
    """

    def __init__(self, margin=128.0, size_average=True):
        super(SoftmaxLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()

    def forward(self, prediction, labels ):

        loss_softmax = self.xentropy(input=prediction, target=labels)

        return  loss_softmax



class TripletSoftmaxLoss(nn.Module):
    """
    Triplet Softmax loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0 ):
        super(TripletSoftmaxLoss, self).__init__()
        self.margin = 0#margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.l2 = nn.PairwiseDistance(p=2)
                    
    def forward(self, anchor, positive, negative, outputs, labels ):
        triplet_positive = 0
        triplet_negative = 0

        loss_softmax = self.loss_fn(input=outputs, target=labels)
        batch_size = anchor.size(0)
        loss_triplet =  torch.zeros(1).cuda()

        for i in range(batch_size):

            triplet_positive = self.l2(anchor[i].view(1,-1), positive[i].view(1,-1))
            triplet_negative = self.l2(anchor[i].view(1,-1), negative[i].view(1,-1))
            distance_triplet = F.relu(triplet_positive - triplet_negative)

            if F.relu(triplet_positive - triplet_negative) > 0:
                loss_triplet += distance_triplet


        loss_total =  loss_softmax  + self.lambda_factor*loss_triplet

        return loss_total, loss_triplet, loss_softmax 



class CentroidsTripletLoss(nn.Module):
    """
    Centroids Triplet Loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, alpha_factor=0.0, beta_factor=0.0, num_classes=0):
        super(CentroidsTripletLoss, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.alpha_factor = alpha_factor
        self.beta_factor = beta_factor
        self.l2 = nn.PairwiseDistance(p=1)
        self.num_classes = num_classes
                    
    def forward(self, anchor, positive, negative, outputs, labels_anchor, labels_neg, exemplars ):
        
        exemplars = exemplars.cuda()
     
        labels = torch.cat((labels_anchor, labels_anchor, labels_neg), 0)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        
        batch_size = anchor.size(0)

        triplet_positive = 0
        triplet_negative = 0

        distance_ref_1 = 0
        distance_neg_1 = 0
        distance_ref_2 = 0
        distance_neg_2 = 0

        loss_center =  torch.zeros(1).cuda()
        loss_triplet =  torch.zeros(1).cuda()
   
        for i in range(batch_size):

            distance_ref = torch.norm(anchor[i] - exemplars[labels_anchor[i].item()], p=2) 
            distance_all = torch.norm(anchor[i] - exemplars, p=2, dim=1)
            
            distance_closest = torch.min(distance_all) 

            forbidden = [1, 2, 10, 11, 13,20, 22, 23, 28, 31, 33, 38, 39, 43, 44, 50]

            if labels_anchor[i] != torch.argmin(distance_all):
                
                if (torch.argmin(distance_all).item() + 1) in forbidden:
                    print( torch.argmin(distance_all).item() + 1)
                    print('error centroid')
                    print(exemplars[torch.argmin(distance_all).item()])
    
            
            triplet_positive =  torch.norm(anchor[i] - positive[i], p=2)
            triplet_negative =  torch.norm(anchor[i] - negative[i], p=2)

            triplet_distance = F.relu(triplet_positive - triplet_negative)
            centroids_distance = F.relu(distance_ref - distance_closest)

            loss_center += centroids_distance 
           
            loss_triplet += triplet_distance


        loss_total =  loss_softmax + self.alpha_factor*loss_triplet  + self.beta_factor*loss_center

        return loss_total, loss_triplet, loss_softmax, loss_center 