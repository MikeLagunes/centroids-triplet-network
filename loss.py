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

            distance_ref_2 = self.l2(anchor[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))
            distance_neg_2 = self.l2(negative[i].view(1,-1), exemplars[labels_neg[i].item()].view(1,-1))

            triplet_positive = self.l2(anchor[i].view(1,-1), positive[i].view(1,-1))
            triplet_negative = self.l2(anchor[i].view(1,-1), negative[i].view(1,-1))

            distance_comp1 = F.relu(distance_ref_1 - distance_neg_1 + self.margin2)
            distance_comp2 = F.relu(distance_neg_2 - distance_ref_2)
            
            distance_triplet = F.relu(triplet_positive - triplet_negative)

            if F.relu(distance_ref_1 - distance_neg_1) > 0:
               # print("Cond 1 not met")
                loss_center += distance_comp1

            if F.relu(distance_neg_2 - distance_ref_2) > 0:
                #print("Cond 2 not met")
                loss_center += distance_comp2

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

            #print('examplars:', exemplars.shape)

            distance_ref = torch.norm(anchor[i] - exemplars[labels_anchor[i].item()], p=2) 
            distance_all = torch.norm(anchor[i] - exemplars, p=2, dim=1)

            #exemplars_temp = exemplars.clone()
            #exemplars_temp[labels_anchor[i].item()] = 10 * torch.ones([512])

            # exemplars_non_self = torch.FloatTensor([exemplars[x].cpu().data.numpy() for x in range(exemplars.shape[0]) if x != labels_anchor[i].item()]).cuda()
            # print('examplars_self_exc:', exemplars_non_self.shape)
            

            #disrance_rest = torch.norm(anchor[i] - exemplars_non_self, p=2, dim=1)
            
            distance_closest = torch.min(distance_all) # argmin
            print('current instance: ', labels_anchor[i], 'closest centroid: ', torch.argmin(distance_all))
            #distance_closest_rest = torch.min(disrance_rest)

            #print('current instance:', labels_anchor[i].item())
            #print('closest:', distance_closest, distance_closest_rest)


            # print('distance_all', distance_all.shape, distance_all)
            
            triplet_positive =  torch.norm(anchor[i] - positive[i], p=2)
            triplet_negative =  torch.norm(anchor[i] - negative[i], p=2)

            triplet_distance = F.relu(triplet_positive - triplet_negative)
            centroids_distance = F.relu(distance_ref - distance_closest)
            #centroids_distance_rest = F.relu(distance_ref - distance_closest_rest)

            #print(centroids_distance_rest)

            #print('triplet distances: ', triplet_distance) #torch.norm(triplet_positive-triplet_negative), p=1)
            #print('centroid distances: ', centroids_distance, centroids_distance_rest)
            #print('centroids ref: ', distance_ref)
            #print('centroids closest: ', torch.min(distance_all), torch.min(disrance_rest), labels_anchor[i].item())

            loss_center += centroids_distance 
           
            loss_triplet += triplet_distance


        loss_total =  loss_softmax + self.alpha_factor*loss_triplet  + self.beta_factor*loss_center##.sum() +

        return loss_total, loss_triplet, loss_softmax, loss_center #