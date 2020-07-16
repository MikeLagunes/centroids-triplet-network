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


labels = {'bottle_oil':0,  'cleaning_toilet':1,  'knife':2,         'shampoo_blue':3,   'toothpaste':4,
'brush':5,       'cup_1':6,            'milk':7,          'shampoo_green':8,  'toy':9,
'can_opener':10,  'cup_2':11,            'potted_plant':12,  'tape':13,           'tv_remote_1':14,
'cereal':15,      'honey':16,            'sauce':17,         'toaster':18,        'tv_remote_2':19}


def ordered_glob(rootdir='.', instances=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
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

    #print ("set:", set_folder) 

    candidates = glob.glob(set_folder + "/*")

    #print ("candidates:", candidates)

    candidates_allowed =[]

    for candidate in candidates:

        for instance in instances:

            if candidate.find(instance) >= 0:

                candidates_allowed.append(candidate)

    #print(candidates_allowed)

    random_index = random.randint(0, len(candidates_allowed)-1)

    #print (len(similar_object_group))
    next_obj = candidates_allowed.pop(random_index)
    next_obj = os.path.split(next_obj)[1]

    next_obj_comp = next_obj[0:next_obj.find('-')]

    print("comparinson",next_obj_comp,obj_curr_comp)

    if next_obj == obj_curr:

        random_index = random.randint(0, len(candidates_allowed)-1)
        next_obj = candidates_allowed.pop(random_index)

    print("next", next_obj)
        #next_obj = os.path.split(next_obj)[1]


    next_imgs = glob.glob(os.path.join(set_folder, next_obj) + "/*")
    next_img = random.choice(next_imgs)

        #print("Next", next_imgs)
        
    # print ("o:",filename)
    # print ("m:",filename[0:-27]+ "%02d" % next_obj + filename[-25:-18] + "%02d" % next_scene + filename[-16:-13] + "%02d_" % next_scene+ "%02d" % next_obj + "_%03d" % next_view + ".png")

    return next_img

def get_different_view(filename):

    obj_folder = os.path.split(filename)[0]

    candidates = glob.glob(obj_folder + "/*")

    #print(obj_folder)

    next_scene = random.choice(candidates)

    #new_filename = filename[0:-18] + "%02d" % next_scene + filename[-16:-13] + "%02d_" % next_scene + "%02d" % obj_num + "_%03d" % next_view + ".png"
          

    return next_scene

def get_label(img_path):

    folder_id = os.path.split(os.path.split(img_path)[0])[1]
    folder_id = folder_id[0:folder_id.find('-')]

    return np.array(labels[folder_id])




class triplet_resnet_household_softmax(data.Dataset):

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
        self.n_classes = 20
        self.n_channels = 3
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base,  instances=instances)

        self.instances = instances

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
        

       

        img_path_similar = get_different_view(img_path)
        img_path_different = get_different_object(img_path, instances=self.instances)

            #print(img_path_similar)
            #print(img_path_different)
   
        img_pos = Image.open(img_path_similar)
        img_neg = Image.open(img_path_different)

        lbl = get_label(img_path)
        lbl_pos =  get_label(img_path_similar)
        lbl_neg =  get_label(img_path_different)


        # lbl =  np.array(labels[folder_id])
        # lbl_pos = np.array(labels[folder_id])
        # lbl_neg = np.array(labels[folder_id])

        #print(img_nby_path)

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
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)


        img_pos = img_pos[:, :, ::-1]
        img_pos = img_pos.astype(np.float64)
        img_pos -= self.mean
        img_pos = m.imresize(img_pos, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img_pos = img_pos.astype(float) / 255.0
        # NHWC -> NCWH
        img_pos = img_pos.transpose(2, 0, 1)



        img_neg = img_neg[:, :, ::-1]
        img_neg = img_neg.astype(np.float64)
        img_neg -= self.mean
        img_neg = m.imresize(img_neg, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img_neg = img_neg.astype(float) / 255.0
        # NHWC -> NCWH
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
    parser.add_argument('--instances', nargs='?', type=str, default='novel',
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

        imgs_nby = imgs_nby.numpy()[:, ::-1, :, :]
        imgs_nby = np.transpose(imgs_nby, [0, 2, 3, 1])

        imgs_pos = imgs_pos.numpy()[:, ::-1, :, :]
        imgs_pos = np.transpose(imgs_pos, [0,2,3,1])

        imgs_pos_nby = imgs_pos_nby.numpy()[:, ::-1, :, :]
        imgs_pos_nby = np.transpose(imgs_pos_nby, [0, 2, 3, 1])

        imgs_neg = imgs_neg.numpy()[:, ::-1, :, :]
        imgs_neg = np.transpose(imgs_neg, [0,2,3,1])

        imgs_neg_nby = imgs_neg_nby.numpy()[:, ::-1, :, :]
        imgs_neg_nby = np.transpose(imgs_neg_nby, [0, 2, 3, 1])

        f, axarr = plt.subplots(bs,6)
        for j in range(bs):      
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(imgs_nby[j])
            axarr[j][2].imshow(imgs_pos[j])
            axarr[j][3].imshow(imgs_pos_nby[j])
            axarr[j][4].imshow(imgs_neg[j])
            axarr[j][5].imshow(imgs_neg_nby[j])

            print(lbl[j], lbl[j], lbl_pos[j], lbl_pos[j], lbl_neg[j], lbl_neg[j])
            
        plt.show()
