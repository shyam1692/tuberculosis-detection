# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 01:05:48 2018

@author: Shyam

Writing code for visualization Gradcam
"""
import os
os.chdir('C:\stuff\Studies\Fall 18\Machine Learning in Signal Processing\Project\Tuberculosis Detection\Code')
data_directory = '../Data/ChinaSet_AllFiles/CXR_png'
import numpy as np
import gradcam_original_unchanged
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

#For VGG16 model, initializing the model, and then loading dictionary
vgg16 = models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./vgg16-397923af.pth"))
num_features = vgg16.classifier[6].in_features
#Removing last layer
features = list(vgg16.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 1)])
features.extend([nn.Sigmoid()])
vgg16.classifier = nn.Sequential(*features)

model_loaded = torch.load("./snapshots_trial/epoch-2-trainerror-0.3693304535637149-testerror-0.22613065326633167.pth", map_location=lambda storage, loc: storage)
vgg16.load_state_dict(model_loaded)

"""
Abhilashs model now
"""
from own_cnn_model import SimpleCNN
model_trained = SimpleCNN()
model_trained.load_state_dict(model_loaded)
#model_trained = model_to_train.to(device)
"""
End abhilash now
"""

#Now we have loaded the model
data_directory = '../Data/ChinaSet_AllFiles/CXR_png'
all_files = os.listdir(data_directory)
del(all_files[len(all_files) - 1])

all_files = np.array(all_files)
for filename in all_files:
    if int(filename[-5]) == 0: 
        continue
    filename_original = data_directory + '/' + filename
    gradcam_original_unchanged.call_gradcam_image(model_trained, filename_original, use_cuda = False, layer_name = '2', filename = filename)
    print(filename)
#filename_original = '../Data/ChinaSet_AllFiles/CXR_png/CHNCXR_0592_1.png'
#gradcam_original_unchanged.call_gradcam_image(vgg16, filename_original, use_cuda = False, layer_name = '30', filename)
import cv2
activations = []
for index in range(0,len(all_files)):
    filename = all_files[index]
    filename_original = data_directory + '/' + filename
    img = cv2.imread(filename_original, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = gradcam_original_unchanged.preprocess_image(img)
    del(img)
    activation = vgg16(input)
    #activations.append(activation.detach().numpy())
    print(filename + ': '+ str(activation.detach().numpy()[0,0]))
    #print(activation.detach().numpy()[0,0])
    
