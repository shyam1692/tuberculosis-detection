# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:54:26 2018

@author: Shyam
All models files
"""
import torch
import torch.nn.functional as F
class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1), 
                                            torch.nn.ReLU(inplace = True), 
                                            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                            torch.nn.Conv2d(18, 20, kernel_size=3, stride=1, padding=1), 
                                            torch.nn.ReLU(inplace = True), 
                                            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                            torch.nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1), 
                                            torch.nn.ReLU(inplace = True), 
                                            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        
        self.classifier = torch.nn.Sequential(torch.nn.Linear(32 * 28 * 28, 64), 
                                              torch.nn.ReLU(inplace = True), 
                                              torch.nn.Linear(64, 1), 
                                              torch.nn.Sigmoid())

    def forward(self, x):
        #Computes the activation of the first convolution
        #Size changes from (3, 224, 224) to (18, 224, 24)
        x = self.features(x)
        
        #Size changes from (3, 224, 224) to (18, 112, 112)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 32 * 28 * 28)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = self.classifier(x)
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        return(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
