"""
Inpainting with Edge-connect inpainting model only


CUDA_VISIBLE_DEVICES=1 python CEEC/l1_adv-infer-inpaint_save.py --cluster "0" --l1_adv
CUDA_VISIBLE_DEVICES=1 python CEEC/l1_adv-infer-inpaint_save.py --cluster "all" --l1_adv 
"""

import os, sys
import numpy as np
import math, PIL
import argparse
from PIL import Image
sys.path.append( '../mars_image_inpainting' )
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from CEEC.models import InpaintingModel, DnCNN
from CEEC.config import Config
from CEEC.dataloader import get_data_loader, get_data_loader_corrupted
from CEEC.metrics import PSNR
from utils import MyHirise
from freq_utils import get_hirise_masked
from tensorboardX import SummaryWriter
from tqdm import tqdm


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

def detect_lines(img, thres=7):
    thres_img = np.asarray((img < thres), dtype='uint8')
    display_img = 255 * thres_img
    return thres_img, display_img

def biggest_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2



parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset", type=str, default="mars_hirise", help="name of the dataset")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--cluster", type=str, default='all', help="which cluster")
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--l1_adv', action='store_true', default=False, help='disable style and perceptual loss')

opt = parser.parse_args()
print(opt)

# cluster = opt.cluster
# prefix = 'cluster_{}_'.format(cluster)

if opt.l1_adv:
    config = Config('./CEEC/config_l1_adv.yml')
    prefix = 'l1_adv_'

from datetime import datetime
now = datetime.now() # current date and time
datetime_f = "_".join(str(now).split())

use_cuda = not opt.no_cuda and torch.cuda.is_available()
torch.manual_seed(1234)
device = torch.device("cuda" if use_cuda else "cpu")

if opt.cluster=='all':
    newdir = "map-proj-v3-l1_adv-all"
    new_root = "/home3/hiya/workspace/data/hirise_dataset/{}/".format(newdir)
else:
    newdir = "map-proj-v3-l1_adv-clustered"
    new_root = "/home3/hiya/workspace/data/hirise_dataset/{}/".format(newdir)
    
#############################################################
if opt.cluster=='all':
    iterator = ['all',]
else:
    iterator = range(5)

for cluster in tqdm(iterator):
    print('##'*30)
    print('Cluster: ', cluster)
    print('##'*30)
    
    opt.cluster=str(cluster)
    prefix = 'cluster_{}_'.format(cluster)
    prefix += 'l1_adv_'
    
    loader_train, test_loader = get_data_loader_corrupted(opt)
    model = InpaintingModel(config, gen_in_channel=2, gen_out_channel=1, disc_in_channel=1).to(device)
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    all_logs = []

    # Initialize generator weights
    if opt.l1_adv:
        PATH = "l1_adv_results/{}{}_generator.h5f".format(prefix, opt.dataset)

 
    print('Loading from ', PATH)
    ckpt = torch.load(PATH, map_location='cpu')
    model.generator.load_state_dict(ckpt)

    # Initialize discriminator weights
    if opt.l1_adv:
        PATH = "l1_adv_results/{}{}_discriminator.h5f".format(prefix, opt.dataset)

    print('Loading from ', PATH)
    ckpt = torch.load(PATH, map_location='cpu')
    model.discriminator.load_state_dict(ckpt)

    model.eval()

    for i, (imgs, filenames) in enumerate(test_loader):   
        h, w = imgs[0,0].numpy().shape
        N = len(imgs)
        all_masks = np.zeros((N, 1, h, w))
        
        for k, img in enumerate(imgs):
            np_img = (255. * (img[0]+1.)/2.).numpy().astype('uint8')
            thres_img, display_img = detect_lines(np_img, thres=7)
            mask = biggest_component(thres_img)
            mask = mask.copy()/255.
            all_masks[k, 0] = 1.0 - mask

        masked_imgs = imgs.type(Tensor) 
        masks = torch.from_numpy(all_masks)
        masks = masks.type(Tensor)                             
        imgs = imgs.type(Tensor) 

        masked_imgs_display = masked_imgs.clone()
        i_outputs, i_gen_loss, i_dis_loss, i_logs = model.process(imgs, masked_imgs, masks)
        outputs_merged = i_outputs

        resized_output = outputs_merged

        for idx in tqdm(range(len(resized_output))):
            x_img = resized_output[idx]
            basename = os.path.basename(filenames[idx])
            newname = os.path.join(new_root, basename)
            save_image((x_img.data + 1.)/2, newname)


    
    
    
