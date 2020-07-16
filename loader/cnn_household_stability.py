import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys
sys.path.append('.')
import matplotlib.pyplot as plt
#matplotlib.use('qt4agg')

from collections import OrderedDict
import os
import numpy as np
import glob
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

def gaussian_noise(image):
    row,col,ch= image.shape
    mean = 0
    #var = 0.5
    sigma = 0.06#var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy


class cnn_household_stability(data.Dataset):

    """ core50 loader 
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
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base, instances=instances)
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

        if self.split == "train":

            folder_id = os.path.split(os.path.split(img_path)[0])[1]
            folder_id = folder_id[0:folder_id.find('-')]

        else:
            folder_id = os.path.split(os.path.split(img_path)[0])[1]

        #print(folder_id)

        lbl =  np.array(labels[folder_id]) #np.array([int(img_path[-10:-8])-1])
        #print(lbl)
        #lbl = np.array([int(img_path[-11:-9])-1]) 

        #img = m.imread(img_path)
        img = Image.open(img_path)
        old_size = img.size

        ratio = float(self.img_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
                    (self.img_size[1]-new_size[1])//2))

        img = np.array(new_im, dtype=np.uint8)

        
        if self.is_transform:
            img, img_w = self.transform(img)
            lbl = self.transform_lbl(lbl)

        return img, img_w, lbl, img_path


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
        img_w = gaussian_noise(img)
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)


        img_w = img_w.transpose(2, 0, 1)


        img = torch.from_numpy(img).float()
        img_w = torch.from_numpy(img_w).float()

        return img, img_w


    def transform_lbl(self, lbl):

        lbl = torch.from_numpy(lbl).long()

        return lbl


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/mikelf/rob/datasets/insitu-household'

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='household',
                        help='Dataset to use [\'tless, core50, toybox etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='full',
                        help='Train Dataset split to use [\'full, known, novel\']')
   

    args = parser.parse_args()
    # All, novel or known splits 
    instances = get_instances(args)

    dst = cnn_household(local_path, is_transform=True, augmentations=None,split="train", 
                 img_size=(224, 224), instances=instances)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    for i, data in enumerate(trainloader):
        imgs, lbls,_ = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs,1)
        for j in range(bs):      
            axarr[j].imshow(imgs[j])
            print(lbls)
        #plt.show()
        plt.pause(1.5)
        #plt.show()

        