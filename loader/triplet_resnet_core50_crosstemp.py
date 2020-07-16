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
from utils import *



test_scenes = ["scene_03","scene_07", "scene_10" ]
train_scenes = ["scene_01"]#,2,4,5,6,8,9,11]

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

                #if folder.find(train_scenes[0]) >= 0 or folder.find(test_scenes[0]) >= 0 or folder.find(test_scenes[1]) >= 0 or folder.find(test_scenes[2]) >= 0 :

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


def get_close_and_far_view(filename, closeness):

    obj_num = int(filename[-10:-8])

    frame_num = int(filename[-7:-4])

    candidates = os.listdir(os.path.split(filename)[0])

    #print(candidates)

    candidates.sort()

    total_candidates = len(candidates)

    index_current = candidates.index(os.path.split(filename)[1])

    if (index_current + closeness + 1) < total_candidates:
        view_close = candidates [index_current + round(closeness/2)]
        view_far = candidates [index_current + closeness]

    else:

        view_close = candidates [index_current - round(closeness/2)]
        view_far = candidates [index_current - closeness]



    #print (obj_num, filename, frame_num, total_candidates, index_current, view_close, view_far)

    filename_close = os.path.join(os.path.split(filename)[0],view_close)
    filename_far = os.path.join(os.path.split(filename)[0],view_far)

    
    return filename_close,filename_far



class triplet_resnet_core50_crosstemp(data.Dataset):

    """tless loader 
    """
   

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(224, 224), augmentations=None, instances=None, closeness=5):
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
        img = Image.open(img_path)

        img_next_path = ""
        

        if self.split[0:5] == "train" :

            img_path_close, img_path_far = get_close_and_far_view(img_path, closeness=self.closeness)

            img_pos = Image.open(img_path_close)

            img_neg = Image.open(img_path_far)


        else:
            
            img_next = Image.open(img_path)
            #/media/mikelf/media_rob/core50_v3/test/obj_01_scene_03/C_03_01_001.png
        
        lbl         = np.array([ int(img_path[-10:-8]) - 1])

        lbl_pos     = np.array([ int(img_path_close[-10:-8]) - 1])

        lbl_neg     = np.array([ int(img_path_far[-10:-8]) - 1])


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
    import torchvision
    import matplotlib.pyplot as plt

    #matplotlib.use('qt4agg')

    parser = argparse.ArgumentParser(description='Hyperparams')
   
    parser.add_argument('--dataset', nargs='?', type=str, default='core50',
                        help='Dataset to use [\'tless, core50, toybox etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='known',
                        help='Train Dataset split to use [\'full, known, novel\']')
    parser.add_argument('--instances_to_eval', nargs='?', type=str, default='all',
                        help='Test Dataset split to use [\'full, known, novel, all\']')
    args = parser.parse_args()


    local_path = '/media/mikelf/rob/datasets/core50_v3'
    instances = get_instances(args)


    t_loader = triplet_resnet_core50_crosstemp(root=local_path, is_transform=True, 
        split='train',
        img_size=(224, 224), 
        augmentations=None, 
        instances=instances)

    print("Found {} images for training".format(len(t_loader.files["train"])))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=2, num_workers=6, shuffle=True)

    
    #dst = triplet_resnet_core50_crosstemp(local_path, split="train", is_transform=True, augmentations=None)
    bs = 2
    #trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
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

            print(lbl[j], lbl[j], lbl_pos[j], lbl_pos[j], lbl_neg[j], lbl_neg[j])
            
       # plt.show()
        plt.pause(1)
        plt.savefig('foo.png')
