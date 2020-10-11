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

        for instance in instances:

            if folder_id.find(instance) >= 0:

                folder_path = folder + "/*"

                filenames_folder = glob.glob(folder_path)
                filenames_folder.sort()
                filenames.extend(filenames_folder)

    return filenames

class cnn_household(data.Dataset):

    """ household loader 
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
        self.files[split] = ordered_glob(rootdir=self.images_base, instances=instances)
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

        if self.split == "train":

            folder_id = os.path.split(os.path.split(img_path)[0])[1]
            folder_id = folder_id[0:folder_id.find('-')]

        else:
            folder_id = os.path.split(os.path.split(img_path)[0])[1]

        lbl =  np.array(labels[folder_id]) #
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
            img, lbl = self.transform(img, lbl)

        return img, lbl, img_path

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
       
        img = img.astype(np.float64)
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        classes = np.unique(lbl)
   
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/path_to/insitu-household'

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

        