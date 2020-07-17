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
    Triplet loss
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


        loss_total =  loss_softmax  + self.lambda_factor*loss_triplet##.sum() +

        return loss_total, loss_triplet, loss_softmax #


class ExemplarSoftmaxLossORG(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0 ):
        super(ExemplarSoftmaxLossORG, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.l2 = nn.PairwiseDistance(p=2)
                    
    def forward(self, anchor, positive, negative, outputs, labels_anchor, labels_neg, exemplars ):
        exemplars = exemplars.cuda()
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative ) #+ self.margin
        labels = torch.cat((labels_anchor, labels_anchor, labels_neg), 0)
        loss_softmax = self.loss_fn(input=self.margin*outputs, target=labels)

        batch_size = anchor.size(0)

        distance_ref_1 = 0
        distance_neg_1 = 0
        distance_ref_2 = 0
        distance_neg_2 = 0

        loss_center =  torch.zeros(1).cuda()

        for i in range(batch_size):
            torch.abs(anchor - positive).sum(1) 

            # distance_ref_1 = torch.abs(anchor[i]- exemplars[labels_anchor[i].item()])
            # distance_neg_1 = torch.abs(negative[i]- exemplars[labels_anchor[i].item()])

            # distance_ref_2 = torch.abs(anchor[i]- exemplars[labels_neg[i].item()])
            # distance_neg_2 = torch.abs(negative[i]- exemplars[labels_neg[i].item()])


            distance_ref_1 = self.l2(anchor[i].view(1,-1), exemplars[labels_anchor[i].item()].view(1,-1))
            distance_neg_1 = self.l2(negative[i].view(1,-1), exemplars[labels_anchor[i].item()].view(1,-1))

            distance_ref_2 = self.l2(anchor[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))
            distance_neg_2 = self.l2(negative[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))


            #distance_ref_1 = self.l2(anchor[i].view(1,-1), exemplars[predicted[i].item()].view(1,-1))
            #distance_neg_1 = self.l2(anchor[i].view(1,-1), exemplars[labels_anchor[i].item()].view(1,-1))  
            #distance_comp1 = F.relu(distance_ref_1 - distance_neg_1)
            #distance_comp2 = F.relu(distance_neg_2 - distance_ref_2)
            # if F.relu(distance_ref_1 - distance_neg_1) > 0:
            #    # print("Cond 1 not met")
            #     loss_center += distance_comp1

            distance_comp1 = F.relu(distance_ref_1 - distance_neg_1)
            distance_comp2 = F.relu(distance_neg_2 - distance_ref_2)

            if F.relu(distance_ref_1 - distance_neg_1) > 0:
               # print("Cond 1 not met")
                loss_center += distance_comp1

            if F.relu(distance_neg_2 - distance_ref_2) > 0:
                #print("Cond 2 not met")
                loss_center += distance_comp2

        #print(loss_center)

        loss_total =  loss_softmax + 1e-3*loss_center + self.lambda_factor*losses.sum()##.sum() +



        # print("Distance to centers 1: ", distance_ref_1.sum(),  distance_neg_1.sum())
        # print("Distance to centers 2: ", distance_ref_2.sum(),  distance_neg_2.sum())
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax, loss_center #


class ExemplarSoftmaxLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, margin2=0.0, size_average=True, lambda_factor=0.0 ):
        super(ExemplarSoftmaxLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.l2 = nn.PairwiseDistance(p=2)
        self.margin2 = margin2
                    
    def forward(self, anchor, positive, negative, outputs, labels_anchor, labels_neg, exemplars ):
        exemplars = exemplars.cuda()
     
        labels = torch.cat((labels_anchor, labels_anchor, labels_neg), 0)
        loss_softmax = self.loss_fn(input=self.margin*outputs, target=labels)
        #loss_softmax = self.loss_fn(input=self.margin*outputs, target=labels)
        
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

            distance_ref_1 = self.l2(anchor[i].view(1,-1), exemplars[labels_anchor[i].item()].view(1,-1))
            distance_neg_1 = self.l2(negative[i].view(1,-1), exemplars[labels_anchor[i].item()].view(1,-1))

            # distance_ref_2 = self.l2(anchor[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))
            # distance_neg_2 = self.l2(negative[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))

            triplet_positive = self.l2(anchor[i].view(1,-1), positive[i].view(1,-1))
            triplet_negative = self.l2(anchor[i].view(1,-1), negative[i].view(1,-1))

            distance_comp1 = F.relu(distance_ref_1 - distance_neg_1 + self.margin2)
           # distance_comp2 = F.relu(distance_neg_2 - distance_ref_2)
            distance_triplet = F.relu(triplet_positive - triplet_negative)

            if F.relu(distance_ref_1 - distance_neg_1) > 0:
               # print("Cond 1 not met")
                loss_center += distance_comp1

            # if F.relu(distance_neg_2 - distance_ref_2) > 0:
            #     #print("Cond 2 not met")
            #     loss_center += distance_comp2

            if F.relu(triplet_positive - triplet_negative) > 0:
                #print("Cond 2 not distance_triplet")
                loss_triplet += distance_triplet

 

        loss_total =  loss_softmax + 1e-2*loss_center + self.lambda_factor*loss_triplet##.sum() +

        return loss_total, loss_triplet, loss_softmax, loss_center #



class PureExemplarSoftmaxLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0):
        super(PureExemplarSoftmaxLoss, self).__init__()
        self.margin = 0
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.l2 = nn.PairwiseDistance(p=1) # p=2
        self.scale = margin

        print(self.scale)
                    
    def forward(self, anchor, negative, outputs, labels_anchor, labels_neg, exemplars ):
        exemplars = exemplars.cuda()
        #distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        #distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        #losses = F.relu(distance_positive - distance_negative + self.margin)
        labels = torch.cat((labels_anchor, labels_neg), 0)
        loss_softmax = self.loss_fn(input=self.scale*outputs, target=labels) # input=3*outputs
        
        batch_size = anchor.size(0)

        distance_ref_1 = 0
        distance_neg_1 = 0
        distance_ref_2 = 0
        distance_neg_2 = 0

        loss_center =  torch.zeros(1).cuda()

        for i in range(batch_size):
           #torch.abs(anchor - positive).sum(1) 

            distance_ref_1 = self.l2(anchor[i].view(1,-1), exemplars[labels_anchor[i].item()].view(1,-1))
            distance_neg_1 = self.l2(negative[i].view(1,-1), exemplars[labels_anchor[i].item()].view(1,-1))

            distance_ref_2 = self.l2(anchor[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))
            distance_neg_2 = self.l2(negative[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))

            distance_comp1 = F.relu(distance_ref_1 - distance_neg_1)
            distance_comp2 = F.relu(distance_neg_2 - distance_ref_2)

            if F.relu(distance_ref_1 - distance_neg_1) > 0:
               # print("Cond 1 not met")
                loss_center += distance_comp1

            if F.relu(distance_neg_2 - distance_ref_2) > 0:
                #print("Cond 2 not met")
                loss_center += distance_comp2

        #print(torch.norm(exemplars))

        if torch.norm(exemplars) == 0:

            loss_total =  loss_softmax #+ self.lambda_factor*loss_center #+ self.lambda_factor*losses.sum()##.sum() +

        else:

            loss_total =  loss_softmax + self.lambda_factor*loss_center


        return loss_total, loss_softmax, loss_center #