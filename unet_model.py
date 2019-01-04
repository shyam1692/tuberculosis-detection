# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:41:03 2018

@author: Shyam
Unet model

Architecture inspired from binary segmentation task
https://github.com/milesial/Pytorch-UNet

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from skimage import io
from skimage.transform import rescale, resize
import torch.utils.data as data
from torch.autograd import Variable
""" Lets define model here"""

class unet(nn.Module):

    def __init__(self):
        super(unet,self).__init__()

        #Input Tensor Dimensions = 256x256x3
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=5,stride=1, padding=2)
        #nn.init.xavier_uniform(self.conv1.weight) #Xaviers Initialisation
        self.activ_1= nn.ReLU()
        #Pooling 1
        self.pool1= nn.MaxPool2d(kernel_size=2, return_indices=True)
        #Output Tensor Dimensions = 128x128x16


        #Input Tensor Dimensions = 128x128x16
        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        #nn.init.xavier_uniform(self.conv2.weight)
        self.activ_2 = nn.ReLU()
        #Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        #Output Tensor Dimensions = 64x64x32

        #Input Tensor Dimensions = 64x64x32
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        #nn.init.xavier_uniform(self.conv3.weight)
        self.activ_3 = nn.ReLU()
        #Output Tensor Dimensions = 64x64x64

        # 32 channel output of pool2 is concatenated
        
        #https://www.quora.com/How-do-you-calculate-the-output-dimensions-of-a-deconvolution-network-layer
        #Input Tensor Dimensions = 64x64x96
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=96,out_channels=32,kernel_size=3,padding=1) ##
        #nn.init.xavier_uniform(self.deconv1.weight)
        self.activ_4=nn.ReLU()
        #UnPooling 1
        self.unpool1=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 128x128x32

        #16 channel output of pool1 is concatenated

        #Input Tensor Dimensions = 128x128x48
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=48,out_channels=16,kernel_size=3,padding=1)
        #nn.init.xavier_uniform(self.deconv2.weight)
        self.activ_5=nn.ReLU()
        #UnPooling 2
        self.unpool2=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 256x256x16

        # 3 Channel input is concatenated

        #Input Tensor Dimensions= 256x256x119
        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=19,out_channels=1,kernel_size=5,padding=2)
        #nn.init.xavier_uniform(self.deconv3.weight)
        self.activ_6=nn.Sigmoid()
        ##Output Tensor Dimensions = 256x256x1
        self.size2 = None
        self.indices2 = None
        self.out_2 = None
        self.size1 = None
        self.indices1 = None
        self.out_1 = None
        self.out_3 = None
    def features(self, x):
        self.out_1 = x
        out = self.conv1(x)
        out = self.activ_1(out)
        self.size1 = out.size()
        out,self.indices1=self.pool1(out)
        self.out_2 = out
        out = self.conv2(out)
        out = self.activ_2(out)
        self.size2 = out.size()
        out, self.indices2=self.pool2(out)
        self.out_3 = out
        out = self.conv3(out)
        out = self.activ_3(out)
        return out
    def classifier(self, out):
        out=torch.cat((out,self.out_3),dim=1)
        out=self.deconv1(out)
        out=self.activ_4(out)
        out=self.unpool1(out,self.indices2,self.size2)
        out=torch.cat((out,self.out_2),dim=1) 
        out=self.deconv2(out)
        out=self.activ_5(out)
        out=self.unpool2(out,self.indices1,self.size1)
        out=torch.cat((out,self.out_1),dim=1)         
        out=self.deconv3(out)
        out=self.activ_6(out)
        out=out
        return out
        
    def forward(self,x):
        out = self.features(x)        
        out = self.classifier(out)
        return out
