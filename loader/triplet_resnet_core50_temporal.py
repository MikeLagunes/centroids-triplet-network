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

from utils import *


train_scenes = [1,2,4,5,6,8,9,11]

train_scene = "scene_01"

def ordered_glob(rootdir='.', instances='', split=''):
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

                # remove for all training scenes

                if folder.find(train_scene) >= 0  : #or folder.find(test_scenes[0]) >= 0 or folder.find(test_scenes[1]) >= 0 or folder.find(test_scenes[2]) >= 0

                    folder_path = folder + "/*"

                    filenames_folder = glob.glob(folder_path)
                    filenames_folder.sort()
                    filenames.extend(filenames_folder)

    #if 'train' in rootdir:
    #    filenames = random.sample(set(filenames), int(0.15 * len(filenames)))

    return filenames


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]


known_classes = [ 3,  4,  5,  6,  7,  8,  9, 12, 14, 15, 16, 17, 18, 19, 21, 24, 25,
       26, 27, 29, 30, 32, 34, 35, 36, 37, 40, 41, 42, 45, 46, 47, 48, 49]

def get_different_object(filename):


    similar_object_group = known_classes[:]#range(1,51) # [ 3,  4,  5,  6,  7,  8,  9, 12, 14, 15, 16, 17, 18, 19, 21, 24, 25, 26, 27, 29, 30, 32, 34, 35, 36, 37, 40, 41, 42, 45, 46, 47, 48, 49]

#range(1,50)#[1,2,3,4,6,11]

    obj_num = int(filename[-10:-8])

    next_view = random.choice(np.arange(50,299))
    next_scene = random.choice(train_scenes)

    random_index = random.randint(0, len(similar_object_group)-1)

    #print (len(similar_object_group))
    next_obj = similar_object_group.pop(random_index)

    if next_obj == obj_num:

        random_index = random.randint(0, len(similar_object_group)-1)

        next_obj = similar_object_group.pop(random_index)
        
    # print ("o:",filename)
    # print ("m:",filename[0:-27]+ "%02d" % next_obj + filename[-25:-18] + "%02d" % next_scene + filename[-16:-13] + "%02d_" % next_scene+ "%02d" % next_obj + "_%03d" % next_view + ".png")

    return filename[0:-27]+ "%02d" % next_obj + filename[-25:-18] + "%02d" % next_scene + filename[-16:-13] + "%02d_" % next_scene + "%02d" % next_obj + "_%03d" % next_view + ".png"


def get_different_view(filename):

    obj_num = int(filename[-10:-8])

    next_scene = random.choice(train_scenes)

    next_view = random.choice(np.arange(15,299))

    new_filename = filename[0:-18] + "%02d" % next_scene + filename[-16:-13] + "%02d_" % next_scene + "%02d" % obj_num + "_%03d" % next_view + ".png"
          

    return new_filename

def get_nby_view(filename, closeness):

    candidates = os.listdir(os.path.split(filename)[0])

    #print(candidates)

    candidates.sort()

    total_candidates = len(candidates)


    index_current = candidates.index(os.path.split(filename)[1])

    if (index_current + closeness) < total_candidates:
        view_close = candidates [random.randint(index_current + 1, index_current + closeness)]

    else:
        view_close = candidates [random.randint(index_current - closeness, index_current - 1)]

    filename_close = os.path.join(os.path.split(filename)[0],view_close)
    
    return filename_close



class triplet_resnet_core50_temporal(data.Dataset):

    """tless loader 
    """
   

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(224, 224), augmentations=None, instances=None, closeness=1):
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
        self.n_classes = 50
        self.n_channels = 3
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base,  instances=instances)

        self.instances = instances
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
        img_path_nby = get_nby_view(img_path, closeness=self.closeness)

        img_path_similar = get_different_view(img_path)
        img_path_similar_nby = get_nby_view(img_path_similar, closeness=self.closeness)

        img_path_different = get_different_object(img_path)
        img_path_different_nby = get_nby_view(img_path_different, closeness=self.closeness)

        img = Image.open(img_path)
        img_nby = Image.open(img_path_nby)

        img_pos = Image.open(img_path_similar)
        img_pos_nby = Image.open(img_path_similar_nby)

        img_neg = Image.open(img_path_different)
        img_neg_nby = Image.open(img_path_different_nby)

        obj_class = os.path.split(os.path.split(img_path)[0])[1]
        obj_class_neg = os.path.split(os.path.split(img_path_different)[0])[1]
  

        #if "hodgepodge" in img_path:
        lbl         = np.array([ int(img_path[-10:-8]) - 1])
        lbl_pos     = np.array([ int(img_path_similar[-10:-8]) - 1])
        lbl_neg     = np.array([ int(img_path_different[-10:-8]) - 1])

        # print(img_nby_path)

        img = self.resize_keepRatio(img)
        img_pos = self.resize_keepRatio(img_pos)
        img_neg = self.resize_keepRatio(img_neg)

        img_nby = self.resize_keepRatio(img_nby)
        img_pos_nby = self.resize_keepRatio(img_pos_nby)
        img_neg_nby = self.resize_keepRatio(img_neg_nby)

        img = np.array(img, dtype=np.uint8)
        img_pos = np.array(img_pos, dtype=np.uint8)
        img_neg = np.array(img_neg, dtype=np.uint8)

        img_nby = np.array(img_nby, dtype=np.uint8)
        img_pos_nby = np.array(img_pos_nby, dtype=np.uint8)
        img_neg_nby = np.array(img_neg_nby, dtype=np.uint8)

        if self.is_transform:
            img = self.transform(img)
            img_pos = self.transform(img_pos)
            img_neg = self.transform(img_neg)

            img_nby = self.transform(img_nby)
            img_pos_nby = self.transform(img_pos_nby)
            img_neg_nby = self.transform(img_neg_nby)


            lbl = self.transform_lbl(lbl)
            lbl_neg = self.transform_lbl(lbl_neg)

        return img, img_nby, img_pos, img_pos_nby, img_neg, img_neg_nby, img_path, lbl, lbl, lbl_neg


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
    import matplotlib.pyplot as plt
    
    local_path = '/media/mikelf/rob/datasets/core50_v3'

    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='core50',
                        help='Dataset to use [\'tless, core50, toybox etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='full',
                        help='Train Dataset split to use [\'full, known, novel\']')

    args = parser.parse_args()
    # All, novel or known splits 
    instances = get_instances(args)

    dst = triplet_resnet_core50_temporal(local_path, is_transform=True, augmentations=None,split="train", 
                 img_size=(224, 224), instances=instances, closeness=5)

    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    f, axarr = plt.subplots(bs,6)
    

    for i, data in enumerate(trainloader): 

        imgs, imgs_nby, imgs_pos, imgs_pos_nby, imgs_neg, imgs_neg_nby, img_path, lbl, lbl, lbl_neg = data
        
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])

        imgs_nby = imgs_nby.numpy()[:, ::-1, :, :]
        imgs_nby = np.transpose(imgs_nby, [0,2,3,1])

        imgs_pos = imgs_pos.numpy()[:, ::-1, :, :]
        imgs_pos = np.transpose(imgs_pos, [0,2,3,1])

        imgs_pos_nby = imgs_pos_nby.numpy()[:, ::-1, :, :]
        imgs_pos_nby = np.transpose(imgs_pos_nby, [0,2,3,1])

        imgs_neg = imgs_neg.numpy()[:, ::-1, :, :]
        imgs_neg = np.transpose(imgs_neg, [0,2,3,1])

        imgs_neg_nby = imgs_neg_nby.numpy()[:, ::-1, :, :]
        imgs_neg_nby = np.transpose(imgs_neg_nby, [0,2,3,1])

        
        for j in range(bs):      
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(imgs_nby[j])
            axarr[j][2].imshow(imgs_pos[j])
            axarr[j][3].imshow(imgs_pos_nby[j])
            axarr[j][4].imshow(imgs_neg[j])
            axarr[j][5].imshow(imgs_neg_nby[j])

            print(lbl[j])
        plt.pause(0.5)
