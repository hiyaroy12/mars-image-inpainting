import os
import os.path
import argparse
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
import torch
from PIL import Image


def product_mask(x, mask):
    return (((x + 1)/2 * mask) - 0.5) / 0.5

hirise_mask_obj = np.load('./data/mask_train_test.npz')
hirise_train_mask = hirise_mask_obj['train']
hirise_test_mask = hirise_mask_obj['test']

def get_hirise_masked(x_im, mask_split='train'): 
    N = len(x_im)
    assert len(x_im[0].shape)==3, "Image should be color" 
    h, w = x_im[0].shape[1:]
    all_masks = np.zeros((N, 1, h, w))
    
    if mask_split=='train':
        sample_mask = hirise_train_mask
    else:
        sample_mask = hirise_test_mask

    for k, mask in enumerate(random.sample(list(sample_mask), N)):
        mask = mask.copy()/255.
        all_masks[k, 0] = 1.0 - mask
    x_masked = product_mask(x_im, all_masks)
    x_masked_parts = product_mask(x_im, (1 - all_masks))
    
    return x_masked, x_masked_parts, all_masks

