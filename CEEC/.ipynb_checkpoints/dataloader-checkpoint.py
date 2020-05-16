# # Dataset loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.celeba import CelebA
import torchvision.transforms as transforms
from CEEC.utils import MyHirise

import glob
import random
import os
import numpy as np
from PIL import Image

use_cuda = torch.cuda.is_available()

def get_data_loader(opt):
    print('Loading dataset ...\n')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if opt.dataset=='mars_hirise':
        dataset_train = MyHirise(split="train", cluster=opt.cluster, kind='clean', test_ratio=0.2,
                       transform=transforms.Compose([transforms.Resize((opt.img_size,opt.img_size)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), 
                                                                          (0.5,))]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

        dataset_val = MyHirise(split="test",  cluster=opt.cluster, kind='clean', test_ratio=0.2,
                                    transform=transforms.Compose([transforms.Resize((opt.img_size,opt.img_size)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5,),
                                                                                      (0.5,))]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)

    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of testing samples: %d\n" % int(len(dataset_val)))
    
    return loader_train, test_loader

#################################################################################################################

def get_data_loader_corrupted(opt):
    print('Loading dataset ...\n')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if opt.dataset=='mars_hirise':
        dataset_train = MyHirise(split="train", cluster=opt.cluster, kind='corrupted', test_ratio=0.2,
                       transform=transforms.Compose([transforms.Resize((opt.img_size,opt.img_size)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), 
                                                                          (0.5,))]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

        dataset_val = MyHirise(split="test",  cluster=opt.cluster, kind='corrupted', test_ratio=0.2,
                                    transform=transforms.Compose([transforms.Resize((256,256)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5,),
                                                                                      (0.5,))]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)
    print("# of testing samples: %d\n" % int(len(dataset_val)))
    return loader_train, test_loader
