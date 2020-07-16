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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.margin = 0
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=4.0, size_average=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
                    #embedding1, embedding2, embedding3, rec1, rec2, rec3, images, images_pos, images_neg
    def forward(self, anchor, positive, negative ):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
    
        return  0.1*losses.sum()

class Stability_loss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0 ):
        super(Stability_loss, self).__init__()
        self.margin = 0
        self.l2 = nn.MSELoss(reduction='sum')
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
                    
    def forward(self, outputs, labels, anchor, anchor_w, ):
        
        loss_l2 = self.l2(anchor, anchor_w)

        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total =  loss_softmax + self.lambda_factor*loss_l2
        #print(loss_total)
        #return losses.mean() if size_average else losses.sum()

        return loss_total, loss_l2, loss_softmax #


class TripletSoftmaxLossORG(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0 ):
        super(TripletSoftmaxLossORG, self).__init__()
        self.margin = 0#margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
                    
    def forward(self, anchor, positive, negative, outputs, labels ):
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total = self.lambda_factor*losses.sum() + loss_softmax
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax #


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





class TripletSoftmaxLossDevNbyTriplet(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """
    # anchor_nby, positive_nby, negative_nby
    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0 ):
        super(TripletSoftmaxLossDevNbyTriplet, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
                    
    def forward(self, anchor, positive, negative, positive_nby, negative_nby, outputs, labels ):
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)

        distance_positive_nby = torch.abs(anchor - positive_nby).sum(1)  # .pow(.5)
        distance_negative_nby = torch.abs(anchor - negative_nby).sum(1)


        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        losses_nby = F.relu(distance_positive_nby - distance_negative_nby)

        #print(losses_nby.sum())
        
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total = self.lambda_factor*losses.sum() + loss_softmax + 1e-3*losses_nby.sum()
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax, losses_nby.sum() #  


class TripletSoftmaxLossDevNbySingle(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """
    # anchor_nby, positive_nby, negative_nby
    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0 ):
        super(TripletSoftmaxLossDevNbySingle, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.l2 = nn.PairwiseDistance(p=2)
        self.margin = 0.1
                    
    def forward(self, anchor, positive, negative, positive_nby, negative_nby, outputs, labels ):
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)

        losses = F.relu(distance_positive - distance_negative)
     
        distance_nearby = self.l2(anchor, positive_nby)
        distances_nearby = F.relu(distance_nearby - self.margin)

        #print(losses_nby.sum())
        
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total = self.lambda_factor*losses.sum() + loss_softmax + 1e-3*distances_nearby.sum()
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax, distance_nearby.sum() # 

class TripletSoftmaxLossNbySingle(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """
    # anchor_nby, positive_nby, negative_nby
    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0 ):
        super(TripletSoftmaxLossNbySingle, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.l2 = nn.PairwiseDistance(p=2)
        print(self.margin)
                    
    def forward(self, anchor, positive, negative, positive_nby, negative_nby, outputs, labels ):
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)

        losses = F.relu(distance_positive - distance_negative)
     
        distance_nearby = self.l2(anchor, negative_nby)
        #distance_nearby = self.l2(anchor, positive_nby)
        distances_nearby = F.relu(distance_nearby - self.margin)

        #print(losses_nby.sum())
        
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total = self.lambda_factor*losses.sum() + loss_softmax + 1e-3*distances_nearby.sum()
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax, distance_nearby.sum()


class TripletTemporalOneLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0,alpha_factor=1e-4 ):
        super(TripletTemporalOneLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.alpha_factor = 1e-4

                    
    def forward(self, anchor, positive, negative, anchor_nby, outputs, labels ):
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        distance_nearby = torch.abs(anchor - anchor_nby).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total = self.lambda_factor*losses.sum() + loss_softmax + self.alpha_factor*distance_nearby.mean()
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax

class TripletTemporalThreeLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0,alpha_factor=1e-4  ):
        super(TripletTemporalThreeLoss, self).__init__()
        self.margin = 0.5
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.alpha_factor = alpha_factor#alpha_factor #4 OK
        self.l2 = nn.PairwiseDistance(p=2)#nn.SmoothL1Loss(reduction='sum')#nn.MSELoss(reduction='sum')
        print(self.alpha_factor, self.lambda_factor)

                    
    def forward(self, anchor, positive, negative, anchor_nby, positive_nby, negative_nby, outputs, labels ):
        #print(torch.norm(anchor[0]),torch.norm(positive[0]),torch.norm(negative[0]))
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        triplets =  torch.cat((anchor, positive, negative), 0)
        triplet_nby = torch.cat((anchor_nby, positive_nby, negative_nby), 0)
        #print(self.l2(anchor[0], triplet_nby[0]))
        #distance_nearby = torch.abs(triplets - triplet_nby).sum(1)
        distance_nearby = self.l2(triplets, triplet_nby)
        #print(distance_nearby)
        distance_nearby = F.relu(distance_nearby - self.margin)# self.l2(triplets, triplet_nby)
        #print(distance_nearby)
        losses = F.relu(distance_positive - distance_negative)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        #loss_total = self.lambda_factor*losses.sum() + loss_softmax + self.alpha_factor*distance_nearby.sum()
        loss_total = self.lambda_factor*losses.sum() + loss_softmax + self.alpha_factor*distance_nearby.sum()
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax#



class TripletTemporalFullLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0,alpha_factor=1e-4  ):
        super(TripletTemporalFullLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.alpha_factor = 1e-4 #4 OK
        self.l2 = nn.MSELoss(reduction='sum')

                    
    def forward(self, anchor, positive, negative, anchor_nby, positive_nby, negative_nby, outputs, labels ):

        anchors = torch.cat((anchor, anchor_nby), 0)
        positives = torch.cat((positive, positive_nby), 0)
        negatives = torch.cat((negative, negative_nby), 0)

        #print(self.l2(anchor, anchor_nby))

        distance_positive = torch.abs(anchors - positives).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchors - negatives).sum(1)  # .pow(.5)

        triplets =  torch.cat((anchor, positive, negative), 0)
        triplet_nby =  torch.cat((anchor_nby, positive_nby, negative_nby), 0)
        distance_nearby = self.l2(triplets, triplet_nby)

        losses = F.relu(distance_positive - distance_negative)
        #print(losses)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        #loss_total = self.lambda_factor*losses.sum() + loss_softmax + self.alpha_factor*distance_nearby.sum()
        loss_total = self.lambda_factor*losses.sum() + loss_softmax + self.alpha_factor*distance_nearby.mean()
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax#

class TripletTemporalLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0):
        super(TripletTemporalLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.l2 = nn.MSELoss(reduction='sum')

                    
    def forward(self, anchor, positive, negative, outputs, labels ):
        distance_ahead = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_before = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        #distance_nearby = torch.abs(triplets - triplet_nby).sum(1)
        #distance_ahead = self.l2(anchor, positive)
        #distance_before = self.l2(anchor, negative)
        #print(distance_nearby)
        #losses = F.relu(distance_ahead.mean() - distance_before.mean())

        losses = F.relu(distance_ahead - distance_before)  
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        #loss_total = self.lambda_factor*losses.sum() + loss_softmax + self.alpha_factor*distance_nearby.sum()
        loss_total = loss_softmax + self.lambda_factor*losses.sum() 
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax


class DoubleTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.0,alpha_factor=1e-4  ):
        super(DoubleTripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor
        self.alpha_factor = 1e-2

                    
    def forward(self, anchor, positive, negative, positive_nby, negative_nby, outputs, labels ):
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative)

        distance_near = torch.abs(negative - positive_nby).sum(1)  # .pow(.5)
        distance_far = torch.abs(negative - negative_nby).sum(1)  # .pow(.5)
        losses_nby = F.relu(distance_near - distance_far)

        loss_softmax = self.loss_fn(input=outputs, target=labels)

        loss_total = self.lambda_factor*losses.sum() + loss_softmax + self.alpha_factor*losses_nby.sum()
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax


class MarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(MarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
