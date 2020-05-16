"""
Inpainting with Edge-connect inpainting model only
#sonic
CUDA_VISIBLE_DEVICES=4 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "0"
CUDA_VISIBLE_DEVICES=6 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "1"
CUDA_VISIBLE_DEVICES=7 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "2"
CUDA_VISIBLE_DEVICES=8 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "3"
CUDA_VISIBLE_DEVICES=9 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "4"
CUDA_VISIBLE_DEVICES=2 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "all" --n_epochs 20

#olimar
CUDA_VISIBLE_DEVICES=2 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "4" --l1_adv
CUDA_VISIBLE_DEVICES=3 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "0" --l1_adv

#lucina
CUDA_VISIBLE_DEVICES=0 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "2" --l1_adv
CUDA_VISIBLE_DEVICES=6 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "all" --l1_adv --n_epochs 20
CUDA_VISIBLE_DEVICES=8 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "1" --l1_adv
CUDA_VISIBLE_DEVICES=9 python CEEC/CEEC-hirise.py --dataset mars_hirise --cluster "3" --l1_adv
"""

import os, sys
import numpy as np
import math, PIL
import argparse
from PIL import Image
sys.path.append( '../mars_image_inpainting' )


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from CEEC.models import InpaintingModel, DnCNN
from CEEC.config import Config
from CEEC.dataloader import get_data_loader #, ImageDataset
from CEEC.metrics import PSNR


# from utils import my_transform, read_file, MyCelebA, MyDTD, MyParis_streetview, MyPlaces2, Myolivetti
# from utils import weights_init_kaiming, batch_PSNR, data_augmentation, create_dir, create_mask, stitch_images, imshow, imsave, Progbar

from utils import MyHirise
# weights_init_kaiming, batch_PSNR, data_augmentation, create_dir, create_mask, stitch_images, imshow, imsave
from freq_utils import get_hirise_masked
# product_mask, make_masked, get_color_images_back, get_color_fft_images, get_color_fft_images_regular, get_color_fft_images_irregular
from tensorboardX import SummaryWriter

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=14, help="size of the batches")
parser.add_argument("--dataset", type=str, default="mars_hirise", help="name of the dataset")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--cluster", type=str, default='all', help="which cluster")
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--l1_adv', action='store_true', default=False, help='disable style and perceptual loss')


opt = parser.parse_args()
print(opt)

cluster = opt.cluster
prefix = 'cluster_{}_'.format(cluster)



if opt.l1_adv:
    config = Config('./CEEC/config_l1_adv.yml')
    prefix += 'l1_adv_'
else:
    config = Config('./CEEC/config.yml')
    
from datetime import datetime
now = datetime.now() # current date and time
datetime_f = "_".join(str(now).split())

os.makedirs("CEEC_only_results/{}{}_images".format(prefix, opt.dataset), exist_ok=True)
# os.makedirs("l1_adv_results/{}{}_images".format(prefix, opt.dataset), exist_ok=True)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
torch.manual_seed(1234)
device = torch.device("cuda" if use_cuda else "cpu")

# PSNR metric
psnr_compute = PSNR(255.0).to(config.DEVICE)

#  Dataloader
loader_train, test_loader = get_data_loader(opt)

# ----------
#  Training
# ----------
model = InpaintingModel(config, gen_in_channel=2, gen_out_channel=1, disc_in_channel=1).to(device)
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
all_logs = []
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(loader_train):   
        
        outputs_irregular = get_hirise_masked(imgs.numpy(), 'train')
        x_masked, x_masked_parts, all_masks = outputs_irregular

        masked_imgs = torch.from_numpy(x_masked).type(Tensor) 
        masks = torch.from_numpy(all_masks)
        masks = masks.type(Tensor)                             
        imgs = imgs.type(Tensor) 
        
        masked_imgs_display = masked_imgs.clone()
#         import ipdb; ipdb.set_trace()
        i_outputs, i_gen_loss, i_dis_loss, i_logs = model.process(imgs, masked_imgs, masks)
        outputs_merged = (i_outputs * (1 - masks)) + (imgs * masks)

        # metrics
        psnr = psnr_compute(postprocess((imgs+1.)/2.), postprocess((outputs_merged+1.)/2.))
        mae = torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(torch.abs(imgs) + torch.abs(outputs_merged)).float()
        
        i_logs.append(('psnr', psnr.item()))
        i_logs.append(('mae', mae.item()))
        
        print("[Epoch %d/%d] [Batch %d/%d]"% (epoch, opt.n_epochs, i, len(loader_train)))
        for log in i_logs:
            print(log[0]+' : '+str(log[1]))
        all_logs.append(i_logs)
        # backward
        model.backward(i_gen_loss, i_dis_loss)
        iteration = model.iteration
               
              
#         import ipdb; ipdb.set_trace()
              
        # Generate sample at sample interval
        batches_done = epoch * len(loader_train) + i
        if batches_done % opt.sample_interval == 0:
            sample = torch.cat((masked_imgs_display.data, outputs_merged.data, imgs.data), -2)
            save_image(sample, "CEEC_only_results/{}{}_images/%d.png".format(prefix, opt.dataset) % batches_done, nrow=8, normalize=True)
            np.save("CEEC_only_results/{}{}_logs.npy".format(prefix, opt.dataset), np.array(all_logs))   
#             save_image(sample, "l1_adv_results/{}{}_images/%d.png".format(prefix, opt.dataset) % batches_done, nrow=8, normalize=True)
#             np.save("l1_adv_results/{}{}_logs.npy".format(prefix, opt.dataset), np.array(all_logs))   
            
    torch.save(model.generator.state_dict(),"CEEC_only_results/{}{}_generator.h5f".format(prefix, opt.dataset))
    torch.save(model.discriminator.state_dict(),"CEEC_only_results/{}{}_discriminator.h5f".format(prefix, opt.dataset))
#     torch.save(model.generator.state_dict(),"l1_adv_results/{}{}_generator.h5f".format(prefix, opt.dataset))
#     torch.save(model.discriminator.state_dict(),"l1_adv_results/{}{}_discriminator.h5f".format(prefix, opt.dataset))
     
    
    
