import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys
sys.path.append('.')
import matplotlib.pyplot as plt

from collections import OrderedDict
import os
import numpy as np
import glob

import random
from utils import *


labels = {'bottle_oil':4,  
'cleaning_toilet':2,  
'knife':8,         
'shampoo_blue':13,   
'toothpaste':16,
'brush':0,       
'cup_1':5,            
'milk':9,          
'shampoo_green':12,  
'toy':17,
'can_opener':1,  
'cup_2':6,            
'potted_plant':10,  
'tape':14,           
'tv_remote_1':18,
'cereal':3,      
'honey':7,            
'sauce':11,         
'toaster':15,        
'tv_remote_2':19}


def ordered_glob(rootdir='.', instances=''):
    """
    """
    filenames = []

    folders = glob.glob(rootdir + "/*")

    for folder in folders:

        folder_id = os.path.split(folder)[1]

        #print (folder_id)

        for instance in instances:

            if folder_id.find(instance) >= 0:

                folder_path = folder + "/*"

                filenames_folder = glob.glob(folder_path)
                filenames_folder.sort()
                filenames.extend(filenames_folder)

    return filenames



def get_different_object(filename, instances=''):

    obj_curr = os.path.split(os.path.split(filename)[0])[1]
    obj_curr_comp = obj_curr[0:obj_curr.find('-')]

    set_folder = os.path.split(os.path.split(filename)[0])[0]
    candidates = glob.glob(set_folder + "/*")

    candidates_allowed =[]

    for candidate in candidates:

        for instance in instances:

            if candidate.find(instance) >= 0:

                candidates_allowed.append(candidate)

    random_index = random.randint(0, len(candidates_allowed)-1)

    next_obj = candidates_allowed.pop(random_index)
    next_obj = os.path.split(next_obj)[1]

    next_obj_comp = next_obj[0:next_obj.find('-')]

    while obj_curr_comp == next_obj_comp:

        random_index = random.randint(0, len(candidates_allowed)-1)
        next_obj = candidates_allowed.pop(random_index)
        next_obj = os.path.split(next_obj)[1]
        next_obj_comp = next_obj[0:next_obj.find('-')]

    next_imgs = glob.glob(os.path.join(set_folder, next_obj) + "/*")
    next_img = random.choice(next_imgs)

    return next_img

def get_different_view(filename):

    obj_folder = os.path.split(filename)[0]

    candidates = glob.glob(obj_folder + "/*")

    next_scene = random.choice(candidates)          

    return next_scene

def get_label(img_path):

    folder_id = os.path.split(os.path.split(img_path)[0])[1]
    folder_id = folder_id[0:folder_id.find('-')]

    return np.array(labels[folder_id])


class triplet_resnet_household_softmax(data.Dataset):

    """household loader 
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
        self.n_classes = 20
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}
        self.images_base = os.path.join(self.root, self.split)
        self.files[split] = ordered_glob(rootdir=self.images_base,  instances=instances)
        self.instances = instances
        self.novel_classes = [5, 12, 17, 1, 10]

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
        img_path_similar = get_different_view(img_path)
        img_path_different = get_different_object(img_path, instances=self.instances)

        img_pos = Image.open(img_path_similar)
        img_neg = Image.open(img_path_different)

        lbl = get_label(img_path)
        lbl_pos =  get_label(img_path_similar)
        lbl_neg =  get_label(img_path_different)

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


    local_path = '/media/mikelf/rob/datasets/insitu-household'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='household',
                        help='Dataset to use [\'tless, core50, toybox etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='full',
                        help='Train Dataset split to use [\'full, known, novel\']')

    args = parser.parse_args()
    # All, novel or known splits 
    instances = get_instances(args)

    dst = triplet_resnet_household_softmax(local_path, is_transform=True, augmentations=None,split="train", 
                 img_size=(224, 224), instances=instances)

    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    for i, data in enumerate(trainloader):

        imgs, imgs_pos, imgs_neg, filenames, lbl, lbl_pos, lbl_neg = data
        
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
