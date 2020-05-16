import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from torchvision.datasets.celeba import CelebA
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os, sys, time, random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets.celeba import CelebA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

def read_file(filename):
    return [line.rstrip('\n') for line in open(filename)]

##############Loading HIRISE dataset###################

class MyHirise(CelebA):
    corrupted_filenames = {'all': './data/allfiles_corrupted.txt',
                            '0': './data/corrupted_files_cluster_0.txt',
                            '1': './data/corrupted_files_cluster_1.txt',
                            '2': './data/corrupted_files_cluster_2.txt',
                            '3': './data/corrupted_files_cluster_3.txt',
                            '4': './data/corrupted_files_cluster_4.txt'}
    
    clean_filenames =     { 'all': './data/allfiles_clean.txt',
                            '0': './data/clean_files_cluster_0.txt',
                            '1': './data/clean_files_cluster_1.txt',
                            '2': './data/clean_files_cluster_2.txt',
                            '3': './data/clean_files_cluster_3.txt',
                            '4': './data/clean_files_cluster_4.txt'}
    
    def __init__(self, cluster='all', kind='clean', test_ratio=0.2, split="train", transform=None): #main file that needs to be modified
        self.split = split
        self.transform = transform
        self.files_dict = {'corrupted': self.corrupted_filenames, 'clean': self.clean_filenames}[kind]
        read_file_name = self.files_dict[cluster]
        all_files = read_file(read_file_name)
        n_files = len(all_files)
        n_train = int((1-test_ratio)*n_files)
        
        if kind=='corrupted':
            self.filenames = all_files
        else:
            if self.split == 'train':
                self.filenames = all_files[:n_train]
            else:
                self.filenames = all_files[n_train:]
 
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.filenames[index])
        if self.transform is not None:
            X = self.transform(X)
        return X, self.filenames[index]
