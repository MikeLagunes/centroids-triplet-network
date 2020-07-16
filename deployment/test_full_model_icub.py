import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

import sys, os, pickle
sys.path.append('models')
sys.path.append('.')
sys.path.append('..')

from embeddings import embeddings

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from torch import optim
import matplotlib.gridspec as gridspec

from loader import get_loader, get_data_path
from models import *
from utils import *
from PIL import Image
import glob
from sklearn.metrics import pairwise_distances

labels_icub = {0:"bodylotion_1"  ,
9:"bodylotion10"  ,
1:"bodylotion2"  ,
2:"bodylotion3"  ,
3:"bodylotion4"  ,
4:"bodylotion5"  ,
5:"bodylotion6"  ,
6:"bodylotion7"    ,
7:"bodylotion8"  ,
8:"bodylotion9"  ,
10:"book_1"  ,
19:"book10"  ,
11:"book2"  ,
12:"book3"  ,
13:"book4"  ,
14:"book5"  ,
15:"book6"  ,
16:"book7"  ,
17:"book8"  ,
18:"book9"  ,
20:"cellphone_1"  ,
29:"cellphone10"  ,
21:"cellphone2"  ,
22:"cellphone3"  ,
23:"cellphone4"  ,
24:"cellphone5"  ,
25:"cellphone6"  ,
26:"cellphone7"  ,
27:"cellphone8"  ,
28:"cellphone9"  ,
30:"flower_1"  ,
39:"flower10"  ,
31:"flower2"  ,
32:"flower3"  ,
33:"flower4"  ,
34:"flower5"  ,
35:"flower6"  ,
36:"flower7"  ,
37:"flower8"  ,
38:"flower9"  ,
40:"glass_1"  ,
49:"glass10"  ,
41:"glass2"  ,
42:"glass3"  ,
43:"glass4"  ,
44:"glass5"  ,
45:"glass6"  ,
46:"glass7"  ,
47:"glass8"  ,
48:"glass9"  ,
50:"hairbrush_1"  ,
59:"hairbrush10"  ,
51:"hairbrush2"  ,
52:"hairbrush3"  ,
53:"hairbrush4"  ,
54:"hairbrush5"  ,
55:"hairbrush6"  ,
56:"hairbrush7"  ,
57:"hairbrush8"  ,
58:"hairbrush9"  ,
60:"hairclip_1"  ,
69:"hairclip10"  ,
61:"hairclip2"  ,
62:"hairclip3"  ,
63:"hairclip4"  ,
64:"hairclip5"  ,
65:"hairclip6"  ,
66:"hairclip7"  ,
67:"hairclip8"  ,
68:"hairclip9"  ,
70:"mouse_1"  ,
79:"mouse10"  ,
71:"mouse2"  ,
72:"mouse3"  ,
73:"mouse4"  ,
74:"mouse5"  ,
75:"mouse6"  ,
76:"mouse7"  ,
77:"mouse8"  ,
78:"mouse9"  ,
80:"mug_1"  ,
89:"mug10"  ,
81:"mug2"  ,
82:"mug3"  ,
83:"mug4"  ,
84:"mug5"  ,
85:"mug6"  ,
86:"mug7"  ,
87:"mug8"  ,
88:"mug9"  ,
90:"ovenglove_1"  ,
99:"ovenglove10"  ,
91:"ovenglove2"  ,
92:"ovenglove3"  ,
93:"ovenglove4"  ,
94:"ovenglove5"  ,
95:"ovenglove6"  ,
96:"ovenglove7"  ,
97:"ovenglove8"  ,
98:"ovenglove9"  ,
100:"pencilcase_1",
109:"pencilcase10",
101:"pencilcase2",
102:"pencilcase3",
103:"pencilcase4",
104:"pencilcase5",
105:"pencilcase6",
106:"pencilcase7",
107:"pencilcase8",
108:"pencilcase9",
110:"perfume_1",
119:"perfume10",
111:"perfume2",
112:"perfume3",
113:"perfume4",
114:"perfume5",
115:"perfume6",
116:"perfume7",
117:"perfume8",
118:"perfume9",
120:"remote_1",
129:"remote10",
121:"remote2",
122:"remote3",
123:"remote4",
124:"remote5",
125:"remote6",
126:"remote7",
127:"remote8",
128:"remote9",
130:"ringbinder_1",
139:"ringbinder10",
131:"ringbinder2",
132:"ringbinder3",
133:"ringbinder4",
134:"ringbinder5",
135:"ringbinder6",
136:"ringbinder7",
137:"ringbinder8",
138:"ringbinder9",
140:"soapdispenser_1",
149:"soapdispenser10",
141:"soapdispenser2",
142:"soapdispenser3",
143:"soapdispenser4",
144:"soapdispenser5",
145:"soapdispenser6",
146:"soapdispenser7",
147:"soapdispenser8",
148:"soapdispenser9",
150:"sodabottle_1",
159:"sodabottle10",
151:"sodabottle2",
152:"sodabottle3",
153:"sodabottle4",
154:"sodabottle5",
155:"sodabottle6",
156:"sodabottle7",
157:"sodabottle8",
158:"sodabottle9",
160:"sprayer_1",
169:"sprayer10",
161:"sprayer2",
162:"sprayer3",
163:"sprayer4",
164:"sprayer5",
165:"sprayer6",
166:"sprayer7",
167:"sprayer8",
168:"sprayer9",
170:"squeezer_1",
179:"squeezer10",
171:"squeezer2",
172:"squeezer3",
173:"squeezer4",
174:"squeezer5",
175:"squeezer6",
176:"squeezer7",
177:"squeezer8",
178:"squeezer9",
180:"sunglasses_1",
189:"sunglasses10",
181:"sunglasses2",
182:"sunglasses3",
183:"sunglasses4",
184:"sunglasses5",
185:"sunglasses6",
186:"sunglasses7",
187:"sunglasses8",
188:"sunglasses9",
190:"wallet_1",
199:"wallet10",
191:"wallet2",
192:"wallet3",
193:"wallet4",
194:"wallet5",
195:"wallet6",
196:"wallet7",
197:"wallet8",
198:"wallet9"}


labels_core = {
0: "plug adapter 1" , 1: "plug adapter 2" , 2: "plug adapter 3" ,3: "plug adapter 4" ,4: "plug adapter 5" ,5: "mobile phone 1",
6: "mobile phone 2",7: "mobile phone 3",8: "mobile phone 4",9: "mobile phone 5",
10:"scissor 1", 11:"scissor 2",12:"scissor 3",13:"scissor 4",14:"scissor 5",
15:"light bulb 1", 16:"light bulb 2",17:"light bulb 3",18:"light bulb 4",19:"light bulb 5",
20:"can 1",21:"can 2",22:"can 3", 23:"can 4",24:"can 5",
25:"glasses 1",26:"glasses 2",27:"glasses 3",28:"glasses 4",29:"glasses 5",
30:"ball 1", 31:"ball 2", 32:"ball 3", 33:"ball 4", 34:"ball 5",
35:"marker 1", 36:"marker 2", 37:"marker 3", 38:"marker 4",39:"marker 5",
40:"cup 1",41:"cup 2", 42:"cup 3", 43:"cup 4", 44:"cup 5",
45:"remote control 1", 46:"remote control 2", 47:"remote control 3",48:"remote control 4", 49:"remote control 5"}


labels_tejani = {
0: "camera" , 1: "coffe cup" , 2: "joystick" ,3: "juice carton" ,4: "milk" ,5: "shampoo"}

labels_toybox = {
0: 'airplane_01', 1: 'airplane_02', 2: 'airplane_03', 3: 'airplane_04', 4: 'airplane_05', 5: 'airplane_06', 6: 'airplane_07', 7: 'airplane_08', 8: 'airplane_09', 9: 'airplane_10', 10: 'airplane_11', 11: 'airplane_12', 12: 'airplane_13', 13: 'airplane_14', 14: 'airplane_15', 15: 'airplane_16', 16: 'airplane_17', 17: 'airplane_18', 18: 'airplane_19', 19: 'airplane_20', 20: 'airplane_21', 21: 'airplane_22', 22: 'airplane_23', 23: 'airplane_24', 24: 'airplane_25', 25: 'airplane_26', 26: 'airplane_27', 27: 'airplane_28', 28: 'airplane_29', 29: 'airplane_30', 30: 'ball_01', 31: 'ball_02', 32: 'ball_03', 33: 'ball_04', 34: 'ball_05', 35: 'ball_06', 36: 'ball_07', 37: 'ball_08', 38: 'ball_09', 39: 'ball_10', 40: 'ball_11', 41: 'ball_12', 42: 'ball_13', 43: 'ball_14', 44: 'ball_15', 45: 'ball_16', 46: 'ball_17', 47: 'ball_18', 48: 'ball_19', 49: 'ball_20', 50: 'ball_21', 51: 'ball_22', 52: 'ball_23', 53: 'ball_24', 54: 'ball_25', 55: 'ball_26', 56: 'ball_27', 57: 'ball_28', 58: 'ball_29', 59: 'ball_30', 60: 'car_01', 61: 'car_02', 62: 'car_03', 63: 'car_04', 64: 'car_05', 65: 'car_06', 66: 'car_07', 67: 'car_08', 68: 'car_09', 69: 'car_10', 70: 'car_11', 71: 'car_12', 72: 'car_13', 73: 'car_14', 74: 'car_15', 75: 'car_16', 76: 'car_17', 77: 'car_18', 78: 'car_19', 79: 'car_20', 80: 'car_21', 81: 'car_22', 82: 'car_23', 83: 'car_24', 84: 'car_25', 85: 'car_26', 86: 'car_27', 87: 'car_28', 88: 'car_29', 89: 'car_30', 90: 'cat_01', 91: 'cat_02', 92: 'cat_03', 93: 'cat_04', 94: 'cat_05', 95: 'cat_06', 96: 'cat_07', 97: 'cat_08', 98: 'cat_09', 99: 'cat_10', 100: 'cat_11', 101: 'cat_12', 102: 'cat_13', 103: 'cat_14', 104: 'cat_15', 105: 'cat_16', 106: 'cat_17', 107: 'cat_18', 108: 'cat_19', 109: 'cat_20', 110: 'cat_21', 111: 'cat_22', 112: 'cat_23', 113: 'cat_24', 114: 'cat_25', 115: 'cat_26', 116: 'cat_27', 117: 'cat_28', 118: 'cat_29', 119: 'cat_30', 120: 'cup_01', 121: 'cup_02', 122: 'cup_03', 123: 'cup_04', 124: 'cup_05', 125: 'cup_06', 126: 'cup_07', 127: 'cup_08', 128: 'cup_09', 129: 'cup_10', 130: 'cup_11', 131: 'cup_12', 132: 'cup_13', 133: 'cup_14', 134: 'cup_15', 135: 'cup_16', 136: 'cup_17', 137: 'cup_18', 138: 'cup_19', 139: 'cup_20', 140: 'cup_21', 141: 'cup_22', 142: 'cup_23', 143: 'cup_24', 144: 'cup_25', 145: 'cup_26', 146: 'cup_27', 147: 'cup_28', 148: 'cup_29', 149: 'cup_30', 150: 'duck_01', 151: 'duck_02', 152: 'duck_03', 153: 'duck_04', 154: 'duck_05', 155: 'duck_06', 156: 'duck_07', 157: 'duck_08', 158: 'duck_09', 159: 'duck_10', 160: 'duck_11', 161: 'duck_12', 162: 'duck_13', 163: 'duck_14', 164: 'duck_15', 165: 'duck_16', 166: 'duck_17', 167: 'duck_18', 168: 'duck_19', 169: 'duck_20', 170: 'duck_21', 171: 'duck_22', 172: 'duck_23', 173: 'duck_24', 174: 'duck_25', 175: 'duck_26', 176: 'duck_27', 177: 'duck_28', 178: 'duck_29', 179: 'duck_30', 180: 'giraffe_01', 181: 'giraffe_02', 182: 'giraffe_03', 183: 'giraffe_04', 184: 'giraffe_05', 185: 'giraffe_06', 186: 'giraffe_07', 187: 'giraffe_08', 188: 'giraffe_09', 189: 'giraffe_10', 190: 'giraffe_11', 191: 'giraffe_12', 192: 'giraffe_13', 193: 'giraffe_14', 194: 'giraffe_15', 195: 'giraffe_16', 196: 'giraffe_17', 197: 'giraffe_18', 198: 'giraffe_19', 199: 'giraffe_20', 200: 'giraffe_21', 201: 'giraffe_22', 202: 'giraffe_23', 203: 'giraffe_24', 204: 'giraffe_25', 205: 'giraffe_26', 206: 'giraffe_27', 207: 'giraffe_28', 208: 'giraffe_29', 209: 'giraffe_30', 210: 'helicopter_01', 211: 'helicopter_02', 212: 'helicopter_03', 213: 'helicopter_04', 214: 'helicopter_05', 215: 'helicopter_06', 216: 'helicopter_07', 217: 'helicopter_08', 218: 'helicopter_09', 219: 'helicopter_10', 220: 'helicopter_11', 221: 'helicopter_12', 222: 'helicopter_13', 223: 'helicopter_14', 224: 'helicopter_15', 225: 'helicopter_16', 226: 'helicopter_17', 227: 'helicopter_18', 228: 'helicopter_19', 229: 'helicopter_20', 230: 'helicopter_21', 231: 'helicopter_22', 232: 'helicopter_23', 233: 'helicopter_24', 234: 'helicopter_25', 235: 'helicopter_26', 236: 'helicopter_27', 237: 'helicopter_28', 238: 'helicopter_29', 239: 'helicopter_30', 240: 'horse_01', 241: 'horse_02', 242: 'horse_03', 243: 'horse_04', 244: 'horse_05', 245: 'horse_06', 246: 'horse_07', 247: 'horse_08', 248: 'horse_09', 249: 'horse_10', 250: 'horse_11', 251: 'horse_12', 252: 'horse_13', 253: 'horse_14', 254: 'horse_15', 255: 'horse_16', 256: 'horse_17', 257: 'horse_18', 258: 'horse_19', 259: 'horse_20', 260: 'horse_21', 261: 'horse_22', 262: 'horse_23', 263: 'horse_24', 264: 'horse_25', 265: 'horse_26', 266: 'horse_27', 267: 'horse_28', 268: 'horse_29', 269: 'horse_30', 270: 'mug_01', 271: 'mug_02', 272: 'mug_03', 273: 'mug_04', 274: 'mug_05', 275: 'mug_06', 276: 'mug_07', 277: 'mug_08', 278: 'mug_09', 279: 'mug_10', 280: 'mug_11', 281: 'mug_12', 282: 'mug_13', 283: 'mug_14', 284: 'mug_15', 285: 'mug_16', 286: 'mug_17', 287: 'mug_18', 288: 'mug_19', 289: 'mug_20', 290: 'mug_21', 291: 'mug_22', 292: 'mug_23', 293: 'mug_24', 294: 'mug_25', 295: 'mug_26', 296: 'mug_27', 297: 'mug_28', 298: 'mug_29', 299: 'mug_30', 300: 'spoon_01', 301: 'spoon_02', 302: 'spoon_03', 303: 'spoon_04', 304: 'spoon_05', 305: 'spoon_06', 306: 'spoon_07', 307: 'spoon_08', 308: 'spoon_09', 309: 'spoon_10', 310: 'spoon_11', 311: 'spoon_12', 312: 'spoon_13', 313: 'spoon_14', 314: 'spoon_15', 315: 'spoon_16', 316: 'spoon_17', 317: 'spoon_18', 318: 'spoon_19', 319: 'spoon_20', 320: 'spoon_21', 321: 'spoon_22', 322: 'spoon_23', 323: 'spoon_24', 324: 'spoon_25', 325: 'spoon_26', 326: 'spoon_27', 327: 'spoon_28', 328: 'spoon_29', 329: 'spoon_30', 330: 'truck_01', 331: 'truck_02', 332: 'truck_03', 333: 'truck_04', 334: 'truck_05', 335: 'truck_06', 336: 'truck_07', 337: 'truck_08', 338: 'truck_09', 339: 'truck_10', 340: 'truck_11', 341: 'truck_12', 342: 'truck_13', 343: 'truck_14', 344: 'truck_15', 345: 'truck_16', 346: 'truck_17', 347: 'truck_18', 348: 'truck_19', 349: 'truck_20', 350: 'truck_21', 351: 'truck_22', 352: 'truck_23', 353: 'truck_24', 354: 'truck_25', 355: 'truck_26', 356: 'truck_27', 357: 'truck_28', 358: 'truck_29', 359: 'truck_30'}

def test(args):
    # Setup image
    # train/eval
    # Setup Dataloader

    root_dir =os.path.split(args.ckpt_path)[0] + "/"

    data_loader = get_loader("cnn_" + args.dataset)
    data_path = get_data_path(args.dataset)
    
    instances = get_instances(args)

    t_loader = data_loader(data_path, is_transform=True, 
        split=args.split,
        img_size=(args.img_rows, args.img_cols), 
        augmentations=None, 
        instances=instances)

    #t_loader = data_loader(data_path, is_transform=True, split=args.split, img_size=(args.img_rows, args.img_cols), augmentations=None)

    n_classes = t_loader.n_classes
    print(n_classes)


    #print("Found {} images in {} split".format(len(t_loader.files[args.split]),args.split ))


    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=False)
     
    # Setup Model

    model = embeddings(pretrained=True,  num_classes=n_classes, ckpt_path=args.ckpt_path, embedding_size=args.embedding_size)
    
    #model.load_state_dict(weights)
    #print ("Model Loaded, Epoch: ", torch.load(args.ckpt_path)['epoch'])

    #print ("Projecting: " + args.dataset + " | " + args.split + " set")

    #print(model)
    
    model = model.cuda()
    model.eval()

    # model.load_state_dict(weights)
    print ("SAE Loaded, Epoch: ", torch.load(args.ckpt_path)['epoch'])

    classifier = pickle.load(open("/home/alexa/Miguel/ste-network/deployment/icub_nc.sav", 'rb'))

    print ("NC classifier Loaded")
    print ("Projecting: " + args.dataset + " | " + args.split + " set")

    # images = Variable(img.cuda(0), volatile=True)
    # gs = gridspec.GridSpec(3, 3)#

    # ax = plt.subplot(gs[:, 0:-1])
    # ax2 = plt.subplot(gs[0, -1])
    # ax3 = plt.subplot(gs[2, -1])

    gs = gridspec.GridSpec(4, 4)#

    ax = plt.subplot(gs[:, 0:-1])
    ax2 = plt.subplot(gs[0:2,-1])
    ax3 = plt.subplot(gs[2:, -1])

    total = 0
    correct = 0

    outputs_prev = np.zeros((1,args.embedding_size))



    for i, (images, labels, path_img) in tqdm(enumerate(trainloader)):
        total +=1
        images = Variable(images.cuda())



        # labels = Variable(labels.cuda())
        outputs = model(images)
        outputs = outputs.cpu()
        outputs = outputs.detach().numpy()

        alpha = 0.95
        distance = np.sum(pairwise_distances(outputs.reshape(1, -1),outputs_prev.reshape(1, -1), metric='euclidean'))
        #print (i, np.sum(pairwise_distances(X_test[i].reshape(1, -1),X_test[i-1].reshape(1, -1), metric='euclidean')))
        if distance < 5.0:
            outputs = ((1-alpha)*outputs+alpha*outputs_prev)


        result = classifier.predict(outputs)
        probs = classifier.predict_proba(outputs)
        best_n = np.argsort(probs, axis=1)[:, -1:]
        labels = labels.cpu()

        outputs_prev = outputs[:]

        

        #print(path_img[0])
        ax.imshow(Image.open(path_img[0]))


        #plt.show()
        ax.set_xticks([])
        ax.set_yticks([])

        #print(best_n, labels.item())

        if best_n == labels.item():
            correct += 1
            color = "green"
        else:
            color = "red"

    #print(correct, total)
       # ax.set_xlabel('Prediction: ' + labels_tejani[int(result)] + "\n Class Probability: %.4f" % probs[0][best_n][0][0], fontsize=16, color=color)
        #ax.set_title('Object Recognition \n ToyBox dataset', fontsize=20, color="blue")
        

        ax.set_title('iCub', fontsize=20, color="blue")

        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel('    Ground truth: \n   ' + str(labels_icub[int(labels)]) ,fontsize=13)


        img_root = glob.glob("/home/alexa/Miguel/dataset_temp/icub/train/{}/*".format(labels_icub[int(labels)]))
        img_root.sort()

        #print(img_root[0])

        ax2.imshow(Image.open(img_root[0]))

        ax3.set_xticks([])
        ax3.set_yticks([])

        if best_n == labels.item():
            color = "green"
        else:
            color = "red"

      
        ax3.set_xlabel('    Object predicted: \n ' + labels_icub[int(result)] + "\n   Class Probability:\n %.2f" % probs[0][best_n][0][0], fontsize=13, color=color)
        
        img_root = glob.glob("/home/alexa/Miguel/dataset_temp/icub/train/{}/*".format(labels_icub[int(result[0])]))
        img_root.sort()

        ax3.imshow(Image.open(img_root[0]))

        
        #ax3.imshow(Image.open("/media/alexa/DATA/Miguel/checkpoints_alexa/ae_cvpr/sae_full_same/toybox/img/" + str(int(result[0] + 1)) + ".png"))

        img_path = os.path.split(path_img[0])[0]

        plt.pause(0.002)

        plt.savefig("/media/alexa/DATA/Miguel/checkpoints_alexa/icra2020/icub/%d.pdf"%i)
        ax.clear()
        ax2.clear()
        ax3.clear()




    # np.savez(root_dir + args.split + "_set_triplet_cnn_softmax_known_" + args.dataset,  embeddings=outputs_embedding, lebels=labels_embedding, filenames=path_imgs)

    print ('Done: ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--id', nargs='?', type=str, default='siamese_cnn_softmax_test', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--ckpt_path', nargs='?', type=str, default='', 
                        help='Path to the saved model')
    parser.add_argument('--test_path', nargs='?', type=str, default='.', 
                        help='Path to saving results')
    parser.add_argument('--dataset', nargs='?', type=str, default='icub', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='deployment',
                        help='Dataset split to use [\'full, known, novel\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, #7 
                        help='Batch Size')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=512, #7 
                        help='Size of the dense layer for inference')
    parser.add_argument('--split', nargs='?', type=str, default='test', 
                        help='Dataset split to use [\'train, eval\']')

    args = parser.parse_args()
    test(args)