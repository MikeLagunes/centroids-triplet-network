import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys
sys.path.append('.')
from collections import OrderedDict
import os
import numpy as np
import glob

import random
import pickle
from utils import *


def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

labels = load_obj("loader/labels_toybox.pkl")

#print (labels)



def ordered_glob(rootdir='.', instances=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []

    folders = glob.glob(rootdir + "/*")

    for folder in folders:

        #if split == 'train':

        folder_id = os.path.split(folder)[1]

        for instance in instances:

            if folder_id.find(instance) >= 0:

        #if folder_id in instances:

                folder_path = folder + "/*"

                filenames_folder = glob.glob(folder_path)
                filenames_folder.sort()
                filenames.extend(filenames_folder)

    return filenames



def get_different_object(filename, instances=''):

    obj_root = os.path.split(os.path.split(filename)[0])[0]

    similar_object_group = instances[:]#known_classes[:]  # os.listdir(obj_root)

    obj_class = os.path.split(os.path.split(filename)[0])[1]

    obj_nxt_index = random.randint(0, len(similar_object_group) - 1)

    obj_next = similar_object_group.pop(obj_nxt_index)


    if obj_next == obj_class: obj_next = similar_object_group.pop(obj_nxt_index - 1)

    obj_next_path = os.path.join(obj_root, obj_next)


    obj_next_views = os.listdir(obj_next_path)

    obj_next_view = random.choice(obj_next_views)

    return os.path.join(obj_next_path, obj_next_view)


def get_different_view(filename):
    obj_folder = os.path.split(filename)[0]

    obj_root = os.path.split(obj_folder)

    obj_item_candidates = os.path.join(obj_root[0], obj_root[1])

    obj_views_candidates = os.listdir(obj_item_candidates)

    random_index = random.randint(0, len(obj_views_candidates) - 1)

    next_view = obj_views_candidates.pop(random_index)

    new_filename = os.path.join(obj_item_candidates, next_view)

    return new_filename


def get_nby_view(filename, closeness):

    candidates = os.listdir(os.path.split(filename)[0])

    #print(candidates)

    candidates.sort()

    total_candidates = len(candidates)

    index_current = candidates.index(os.path.split(filename)[1])

    if (index_current + closeness) < total_candidates:
        view_close = candidates [index_current + closeness]

    else:
        view_close = candidates [index_current - closeness]

    filename_close = os.path.join(os.path.split(filename)[0],view_close)
    
    return filename_close

def get_close_and_far_view(filename, closeness):

    #print(filename,closeness)


    candidates = os.listdir(os.path.split(filename)[0])

    #print(candidates)

    candidates.sort()

    total_candidates = len(candidates)

    index_current = candidates.index(os.path.split(filename)[1])

    if (index_current + closeness + 1) < total_candidates:
        view_close = candidates [index_current + 1]#candidates [index_current + round(closeness/2)]
        view_far = candidates [index_current + closeness]

    else:

        view_close = candidates [index_current - 1]#candidates [index_current - round(closeness/2)]
        view_far = candidates [index_current - closeness]

    filename_close = os.path.join(os.path.split(filename)[0],view_close)
    filename_far = os.path.join(os.path.split(filename)[0],view_far)

    
    return filename_close,filename_far


class double_triplet_resnet_toybox(data.Dataset):
    """tless loader
    """

    def __init__(self, root, split="train", is_transform=False,
                 img_size=(224, 224), augmentations=None, instances ='', closeness=1):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 360
        self.n_channels = 3
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}
        self.instances = instances
        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base, instances=self.instances)
        self.closeness = closeness

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        img_path_similar = get_different_view(img_path)
        img_path_different = get_different_object(img_path, instances = self.instances)
        img_path_close, img_path_far = get_close_and_far_view(img_path_different, closeness=self.closeness)


        img = Image.open(img_path)
        img_pos = Image.open(img_path_similar)
        img_neg = Image.open(img_path_different)
        img_close = Image.open(img_path_close)
        img_far = Image.open(img_path_far)

        obj_class = os.path.split(os.path.split(img_path)[0])[1]
        obj_class_neg = os.path.split(os.path.split(img_path_different)[0])[1]
  

        #if "hodgepodge" in img_path:
        lbl = np.array([labels[obj_class]])
        lbl_neg = np.array([labels[obj_class_neg]])


        # print(img_nby_path)

        img = self.resize_keepRatio(img)
        img_pos = self.resize_keepRatio(img_pos)
        img_neg = self.resize_keepRatio(img_neg)

        img_close = self.resize_keepRatio(img_close)
        img_far = self.resize_keepRatio(img_far)

        img = np.array(img, dtype=np.uint8)
        img_pos = np.array(img_pos, dtype=np.uint8)
        img_neg = np.array(img_neg, dtype=np.uint8)
        img_close = np.array(img_close, dtype=np.uint8)
        img_far = np.array(img_far, dtype=np.uint8)

        if self.is_transform:
            img = self.transform(img)
            img_pos = self.transform(img_pos)
            img_neg = self.transform(img_neg)
            img_close = self.transform(img_close)
            img_far = self.transform(img_far)

            lbl = self.transform_lbl(lbl)
            lbl_neg = self.transform_lbl(lbl_neg)


        return img, img_pos, img_neg, img_close, img_far, img_path, lbl, lbl, lbl_neg


    def resize_keepRatio(self, img):

        old_size = img.size

        ratio = float(self.img_size[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0] - new_size[0]) // 2,
                           (self.img_size[1] - new_size[1]) // 2))

        return new_im

    def transform(self, img):
        """transform

        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0

        img = img.astype(float) / 255.0
      
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
       

        return img

    def transform_lbl(self, lbl):

        lbl = torch.from_numpy(lbl).long()

        return lbl


if __name__ == '__main__':
    import torchvision
    #%matplotlib inline
    import matplotlib.pyplot as plt
    #matplotlib.use('TkAgg')
    #plt.ion()
    #import argparse

    local_path = '/home/mikelf/Datasets/toybox'

    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='toybox',
                        help='Dataset to use [\'tless, core50, toybox etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='full',
                        help='Train Dataset split to use [\'full, known, novel\']')

    args = parser.parse_args()
    # All, novel or known splits 
    instances = get_instances(args)

    dst = double_triplet_resnet_toybox(local_path, is_transform=True, augmentations=None,split="train", 
                 img_size=(224, 224), instances=instances, closeness=20)

    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    f, axarr = plt.subplots(bs,5)
    

    for i, data in enumerate(trainloader): 

        imgs, imgs_pos, imgs_neg, img_close, img_far, img_path, lbl, lbl, lbl_neg = data
        
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])

        imgs_pos = imgs_pos.numpy()[:, ::-1, :, :]
        imgs_pos = np.transpose(imgs_pos, [0,2,3,1])

        imgs_neg = imgs_neg.numpy()[:, ::-1, :, :]
        imgs_neg = np.transpose(imgs_neg, [0,2,3,1])

        img_close = img_close.numpy()[:, ::-1, :, :]
        img_close = np.transpose(img_close, [0,2,3,1])

        img_far = img_far.numpy()[:, ::-1, :, :]
        img_far = np.transpose(img_far, [0,2,3,1])

      
        
        for j in range(bs):      
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(imgs_pos[j])
            axarr[j][2].imshow(imgs_neg[j])
            axarr[j][3].imshow(img_close[j])
            axarr[j][4].imshow(img_far[j])
            #axarr[j][5].imshow(imgs_neg_nby[j])

            print(lbl[j])
        plt.pause(0.5)

        #plt.draw()    
        
