#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:14:47 2019

@author: viswanatha
"""

import torch
from net import *
import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser()
    
#parser.add_argument('model_path',help='Path to the trained model')
parser.add_argument('image_path',help='Path to test image')
#parser.add_argument('use_gpu', help='True if gpu is available')
args = parser.parse_args()
    
def load_network(network):
    #save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    save_path = 'checkpoints/market/resnet50/net_last.pth'
    network.load_state_dict(torch.load(save_path))
    return network


model_dict = torch.load('checkpoints/market/resnet50/net_last.pth')

num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}

num_cls = num_cls_dict['market']
model = model_dict['resnet50'](num_cls)

model = load_network(model)

img = cv2.imread(args.image_path)
im = np.moveaxis(img, -1, 0)
im = np.expand_dims(im, axis=0)

labels = ["young", "teenager", "adult", "old",
          "backpack", "bag", "handbag",    
          "clothes", "down", "up", "hair",    
          "hat",   "gender",  
          "upblack","upwhite", 
          "upred", "uppurple", "upyellow", "upgray", 
          "upblue", "upgreen",
          "downblack", "downwhite",
          "downpink", "downpurple", 
          "downyellow", "downgray", 
          "downblue", "downgreen", "downbrown"]

img1 = torch.Tensor(im)
model.eval()
outs = model.forward(img1)
for index in range(30):
    if outs[:,index]==1:
        print (labels[index])
            
