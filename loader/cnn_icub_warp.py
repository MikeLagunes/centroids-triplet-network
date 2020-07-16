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

import cv2
import random
from utils import *

labels = {"bodylotion_1":0,
"bodylotion10":9,
"bodylotion2":1,
"bodylotion3":2,
"bodylotion4":3,
"bodylotion5":4,
"bodylotion6":5,
"bodylotion7":6,
"bodylotion8":7,
"bodylotion9":8,
"book_1":10,
"book10":19,
"book2":11,
"book3":12,
"book4":13,
"book5":14,
"book6":15,
"book7":16,
"book8":17,
"book9":18,
"cellphone_1":20,
"cellphone10":29,
"cellphone2":21,
"cellphone3":22,
"cellphone4":23,
"cellphone5":24,
"cellphone6":25,
"cellphone7":26,
"cellphone8":27,
"cellphone9":28,
"flower_1":30,
"flower10":39,
"flower2":31,
"flower3":32,
"flower4":33,
"flower5":34,
"flower6":35,
"flower7":36,
"flower8":37,
"flower9":38,
"glass_1":40,
"glass10":49,
"glass2":41,
"glass3":42,
"glass4":43,
"glass5":44,
"glass6":45,
"glass7":46,
"glass8":47,
"glass9":48,
"hairbrush_1":50,
"hairbrush10":59,
"hairbrush2":51,
"hairbrush3":52,
"hairbrush4":53,
"hairbrush5":54,
"hairbrush6":55,
"hairbrush7":56,
"hairbrush8":57,
"hairbrush9":58,
"hairclip_1":60,
"hairclip10":69,
"hairclip2":61,
"hairclip3":62,
"hairclip4":63,
"hairclip5":64,
"hairclip6":65,
"hairclip7":66,
"hairclip8":67,
"hairclip9":68,
"mouse_1":70,
"mouse10":79,
"mouse2":71,
"mouse3":72,
"mouse4":73,
"mouse5":74,
"mouse6":75,
"mouse7":76,
"mouse8":77,
"mouse9":78,
"mug_1":80,
"mug10":89,
"mug2":81,
"mug3":82,
"mug4":83,
"mug5":84,
"mug6":85,
"mug7":86,
"mug8":87,
"mug9":88,
"ovenglove_1":90,
"ovenglove10":99,
"ovenglove2":91,
"ovenglove3":92,
"ovenglove4":93,
"ovenglove5":94,
"ovenglove6":95,
"ovenglove7":96,
"ovenglove8":97,
"ovenglove9":98,
"pencilcase_1":100,
"pencilcase10":109,
"pencilcase2":101,
"pencilcase3":102,
"pencilcase4":103,
"pencilcase5":104,
"pencilcase6":105,
"pencilcase7":106,
"pencilcase8":107,
"pencilcase9":108,
"perfume_1":110,
"perfume10":119,
"perfume2":111,
"perfume3":112,
"perfume4":113,
"perfume5":114,
"perfume6":115,
"perfume7":116,
"perfume8":117,
"perfume9":118,
"remote_1":120,
"remote10":129,
"remote2":121,
"remote3":122,
"remote4":123,
"remote5":124,
"remote6":125,
"remote7":126,
"remote8":127,
"remote9":128,
"ringbinder_1":130,
"ringbinder10":139,
"ringbinder2":131,
"ringbinder3":132,
"ringbinder4":133,
"ringbinder5":134,
"ringbinder6":135,
"ringbinder7":136,
"ringbinder8":137,
"ringbinder9":138,
"soapdispenser_1":140,
"soapdispenser10":149,
"soapdispenser2":141,
"soapdispenser3":142,
"soapdispenser4":143,
"soapdispenser5":144,
"soapdispenser6":145,
"soapdispenser7":146,
"soapdispenser8":147,
"soapdispenser9":148,
"sodabottle_1":150,
"sodabottle10":159,
"sodabottle2":151,
"sodabottle3":152,
"sodabottle4":153,
"sodabottle5":154,
"sodabottle6":155,
"sodabottle7":156,
"sodabottle8":157,
"sodabottle9":158,
"sprayer_1":160,
"sprayer10":169,
"sprayer2":161,
"sprayer3":162,
"sprayer4":163,
"sprayer5":164,
"sprayer6":165,
"sprayer7":166,
"sprayer8":167,
"sprayer9":168,
"squeezer_1":170,
"squeezer10":179,
"squeezer2":171,
"squeezer3":172,
"squeezer4":173,
"squeezer5":174,
"squeezer6":175,
"squeezer7":176,
"squeezer8":177,
"squeezer9":178,
"sunglasses_1":180,
"sunglasses10":189,
"sunglasses2":181,
"sunglasses3":182,
"sunglasses4":183,
"sunglasses5":184,
"sunglasses6":185,
"sunglasses7":186,
"sunglasses8":187,
"sunglasses9":188,
"wallet_1":190,
"wallet10":199,
"wallet2":191,
"wallet3":192,
"wallet4":193,
"wallet5":194,
"wallet6":195,
"wallet7":196,
"wallet8":197,
"wallet9":198}

def ordered_glob(rootdir='.', instances=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []

    folders = glob.glob(rootdir + "/*")

    print(sum([len(files) for r, d, files in os.walk(rootdir)]))

    #print(len(folders))

    for folder in folders:

        #if split == 'train':

        folder_id = os.path.split(folder)[1]

        for instance in instances:

            #print (instance,folder_id )

            if folder_id.find(instance) >= 0:

                # remove for all training scenes

                folder_path = folder + "/*"

                filenames_folder = glob.glob(folder_path)
                filenames_folder.sort()
                filenames.extend(filenames_folder)

    return filenames


def get_label(img_path):

    folder_id = os.path.split(os.path.split(img_path)[0])[1]
    #folder_id = folder_id[0:folder_id.find('-')]

    return np.array(labels[folder_id])


def warp_image(image):

    augmentation= random.randint(1,4)

    if augmentation ==1: 
        scale_factor = random.uniform(0.97, 1.03)
        image = cv2.resize(image,None,fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
        image = cv2.resize(image,(224, 224), interpolation = cv2.INTER_CUBIC)
        return image

    elif augmentation ==2:
        rows,cols,_ = image.shape
        M = np.float32([[1,0,random.randint(-2,2)],[0,1,random.randint(-2,2)]])
        image = cv2.warpAffine(image,M,(cols,rows))
    elif augmentation ==3:
        rows,cols,_ = image.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-1,1),1)
        image = cv2.warpAffine(image,M,(cols,rows))
    elif augmentation ==4:

        rows,cols,_ = image.shape
        M = np.float32([[1,0,random.randint(-2,2)],[0,1,random.randint(-2,2)]])
        image = cv2.warpAffine(image,M,(cols,rows))
        M = cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-1,1),1)
        image = cv2.warpAffine(image,M,(cols,rows))

        angle = 0
        translation = 0


    return image


class cnn_icub_warp(data.Dataset):

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
        self.n_classes = 200
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

            # /media/mikelf/media_rob/core50_v3/test/obj_01_scene_03/C_03_01_001.png

        lbl = get_label(img_path)
        img = self.resize_keepRatio(img)
        img = np.array(img, dtype=np.uint8)
      

        #print(lbl, lbl_pos, lbl_neg)

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
        img_w = warp_image(img)
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

    local_path = '/media/mikelf/rob/datasets/icub_ml'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='icub',
                        help='Dataset to use [\'tless, core50, toybox etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='full',
                        help='Train Dataset split to use [\'full, known, novel\']')

    args = parser.parse_args()
    # All, novel or known splits 
    instances = get_instances(args)

    dst = cnn_icub(local_path, is_transform=True, augmentations=None,split="train", 
                 img_size=(224, 224), instances=instances)

    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    for i, data in enumerate(trainloader):

        imgs, lbl, filenames = data
        
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
    
        f, axarr = plt.subplots(bs,1)
        for j in range(bs):      
            axarr[j].imshow(imgs[j])
            print(lbl[j])
            
        plt.show()
