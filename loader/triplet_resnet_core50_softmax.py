import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys

from collections import OrderedDict
import os
import numpy as np
import glob
import random
import argparse

train_scenes = [1]

train_scene = "scene_01"

def ordered_glob(rootdir='.', instances='', split=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []

    folders = glob.glob(rootdir + "/*")

    for folder in folders:

        folder_id = os.path.split(folder)[1]

        for instance in instances:

            if folder_id.find(instance) >= 0:

                if folder.find(train_scene) >= 0: 

                    folder_path = folder + "/*"

                    filenames_folder = glob.glob(folder_path)
                    filenames_folder.sort()
                    filenames.extend(filenames_folder)

    return filenames


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

# known clases range [1, 50]

known_classes = [ 3,  4,  5,  6,  7,  8,  9, 12, 14, 15, 16, 17, 18, 19, 21, 24, 25,
       26, 27, 29, 30, 32, 34, 35, 36, 37, 40, 41, 42, 45, 46, 47, 48, 49]

def get_different_object(filename):


    similar_object_group = known_classes[:]

    obj_num = int(filename[-10:-8])

    next_view = random.choice(np.arange(50,299))
    next_scene = random.choice(train_scenes)

    random_index = random.randint(0, len(similar_object_group)-1)

    next_obj = similar_object_group.pop(random_index)

    if next_obj == obj_num:

        random_index = random.randint(0, len(similar_object_group)-1)

        next_obj = similar_object_group.pop(random_index)
    
    return filename[0:-27]+ "%02d" % next_obj + filename[-25:-18] + "%02d" % next_scene + filename[-16:-13] + "%02d_" % next_scene + "%02d" % next_obj + "_%03d" % next_view + ".png"


def get_different_view(filename):

    obj_num = int(filename[-10:-8])

    next_scene = random.choice(train_scenes)

    next_view = random.choice(np.arange(15,299))

    new_filename = filename[0:-18] + "%02d" % next_scene + filename[-16:-13] + "%02d_" % next_scene + "%02d" % obj_num + "_%03d" % next_view + ".png"
          
    return new_filename



class triplet_resnet_core50_softmax(data.Dataset):

    """tless loader 
    """
   

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(224, 224), augmentations=None, instances=None):
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
        self.files = {}
        self.images_base = os.path.join(self.root, self.split)
        self.files[split] = ordered_glob(rootdir=self.images_base,  instances=instances)
        self.instances = instances
        self.novel_classes = [0, 1, 9, 10, 12, 19, 21, 22, 27, 30, 32, 37, 38, 42, 43, 49]

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
        img = Image.open(img_path)

        img_next_path = ""
        
        if self.split[0:5] == "train" :

            img_path_similar = get_different_view(img_path)
            img_path_different = get_different_object(img_path)
            img_pos = Image.open(img_path_similar)
            img_neg = Image.open(img_path_different)


        else:
            
            img_next = Image.open(img_path)
        
        lbl         = np.array([ int(img_path[-10:-8]) - 1])
        lbl_pos     = np.array([ int(img_path_similar[-10:-8]) - 1])
        lbl_neg     = np.array([ int(img_path_different[-10:-8]) - 1])

        img = self.resize_keepRatio(img)
        img_pos = self.resize_keepRatio(img_pos)
        img_neg = self.resize_keepRatio(img_neg)

        img = np.array(img, dtype=np.uint8)
        img_pos = np.array(img_pos, dtype=np.uint8)
        img_neg = np.array(img_neg, dtype=np.uint8)

        if self.is_transform:
            img, img_pos, img_neg = self.transform(img, img_pos, img_neg )
            lbl, lbl_pos, lbl_neg = self.transform_lbl(lbl, lbl_pos, lbl_neg)

        return img, img_pos, img_neg, img_path, lbl, lbl_pos, lbl_neg

    def resize_keepRatio(self, img):

        old_size = img.size

        ratio = float(self.img_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
                    (self.img_size[1]-new_size[1])//2))

        return new_im

    def transform(self, img, img_pos, img_neg):
        """transform

        :param img:
        :param lbl:
        """

        img = img.astype(np.float64)
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        img_pos = img_pos.astype(np.float64)
        img_pos = img_pos.astype(float) / 255.0
        img_pos = img_pos.transpose(2, 0, 1)

        img_neg = img_neg.astype(np.float64)
        img_neg = img_neg.astype(float) / 255.0
        img_neg = img_neg.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        img_pos = torch.from_numpy(img_pos).float()
        img_neg = torch.from_numpy(img_neg).float()
     
        return img, img_pos, img_neg


    def transform_lbl(self, lbl, lbl_pos, lbl_neg):

        lbl = torch.from_numpy(lbl).long()
        lbl_pos = torch.from_numpy(lbl_pos).long()
        lbl_neg = torch.from_numpy(lbl_neg).long()

        return lbl, lbl_pos, lbl_neg



if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    sys.path.append('utils')
    sys.path.append('.')
    sys.path.append('..')

    from utils import get_instances


    local_path = '/home/mikelf/datasets/core50'
  
    bs = 2

    parser = argparse.ArgumentParser(description='Hyperparams')

    parser.add_argument('--dataset', nargs='?', type=str, default='core50',
                        help='Dataset to use [\'tless, core50, toybox etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--instances', nargs='?', type=str, default='known',
                        help='Train Dataset split to use [\'full, known, novel\']')

    args = parser.parse_args()

    # All, novel or known splits 
    instances = get_instances(args)

    
    t_loader = triplet_resnet_core50_softmax(local_path, is_transform=True, 
        split='train',
        img_size=(args.img_rows, args.img_cols), 
        augmentations=None, 
        instances=instances)

    trainloader = data.DataLoader(t_loader, batch_size=bs, num_workers=6, shuffle=True)

    for i, data in enumerate(trainloader):
        imgs, imgs_pos, imgs_neg, filenames, lbl, lbl_pos, lbl_neg = data

        ##print(imgs.shape)
        
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])

        imgs_pos = imgs_pos.numpy()[:, ::-1, :, :]
        imgs_pos = np.transpose(imgs_pos, [0,2,3,1])

        imgs_neg = imgs_neg.numpy()[:, ::-1, :, :]
        imgs_neg = np.transpose(imgs_neg, [0,2,3,1])


        f, axarr = plt.subplots(bs,3)
        for j in range(bs):      
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(imgs_pos[j])
            axarr[j][2].imshow(imgs_neg[j])
            print(lbl[j], lbl_pos[j], lbl_neg[j])
            
        plt.show()
