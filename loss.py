from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class CentroidsTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, alpha_factor=0.0, beta_factor=0.0, num_classes=0):
        super(CentroidsTripletLoss, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.alpha_factor = alpha_factor
        self.beta_factor = beta_factor
        self.num_classes = num_classes
                    
    def forward(self, anchor, positive, negative, outputs, labels_anchor, labels_neg, exemplars):
        
        exemplars = exemplars.cuda()
     
        labels = torch.cat((labels_anchor, labels_anchor, labels_neg), 0)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        
        batch_size = anchor.size(0)

        loss_center =  torch.zeros(1).cuda()
        loss_triplet =  torch.zeros(1).cuda()
   
        for i in range(batch_size):

            distance_ref = torch.norm(anchor[i] - exemplars[labels_anchor[i].item()], p=2) 
            distance_all = torch.norm(anchor[i] - exemplars, p=2, dim=1)
            
            distance_closest = torch.min(distance_all)
              
            triplet_positive =  torch.norm(anchor[i] - positive[i], p=2)
            triplet_negative =  torch.norm(anchor[i] - negative[i], p=2)

            triplet_distance = F.relu(triplet_positive - triplet_negative)
            centroids_distance = F.relu(distance_ref - distance_closest)

            loss_center += centroids_distance 
            loss_triplet += triplet_distance


        loss_total = loss_softmax + self.alpha_factor*loss_center  + self.beta_factor*loss_triplet 

        return loss_total, loss_triplet, loss_softmax, loss_center 