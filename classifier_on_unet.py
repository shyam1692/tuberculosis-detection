# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:46:34 2018

@author: Shyam
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:54:26 2018

@author: Shyam
All models files
"""
import torch
import torch.nn.functional as F
class ClassifierOnUnet(torch.nn.Module):

    def __init__(self):
        super(ClassifierOnUnet, self).__init__()
        #Input channels = 3, output channels = 18
        self.classifier = torch.nn.Sequential(torch.nn.Linear(64*64*64, 64), 
                                              torch.nn.ReLU(inplace = True), 
                                              torch.nn.Linear(64, 1), 
                                              torch.nn.Sigmoid())

    def forward(self, x):
        #x = x.view(-1, x.size(0))
        x = x.view(-1, 64*64*64)
        
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
