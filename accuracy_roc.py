# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:30:41 2018

@author: Shyam

Accuracy Calculation
"""

import os
os.chdir('C:\stuff\Studies\Fall 18\Machine Learning in Signal Processing\Project\Tuberculosis Detection\Code')


import numpy as np
#Checking unique image sizes of all images, and seeing if we need to resize or not.

#os.listdir(data_directory)
#Testing on montgomery dataset
data_directory = '../Data/MontgomerySet/CXR_png'

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

test_files = os.listdir(data_directory)
del(test_files[len(test_files) - 1])

data_transforms = transforms.Compose([ transforms.ToTensor()])

def CNNloader(data_root, filename):
    filename_actual = data_root + '/' + filename
    data_old = io.imread(filename_actual)
    data_old = resize(data_old,(224,224))
    data_old = np.array(data_old, dtype=np.float32)
    if len(data_old.shape) <= 2:
        data_new = np.zeros(data_old.shape + (3,))
        data_new[:,:,0] = np.array(data_old)
        data_new[:,:,1] = np.array(data_old)
        data_new[:,:,2] = np.array(data_old)
        data_old = np.array(data_new, dtype=np.float32)
        del(data_new)
    data_old = data_transforms(np.array(data_old))
    #data = (data-0.5)/0.5    
    #data_old = data_old[np.newaxis, ...]
    return data_old

#Here, we have getitem function according to index, which will let the data be accessed as index
#This will be required by Pytorch data loader
#Also, we would need to change the length function
#Basically, we are defining a way here as on how to read dataset

#In init function, we will be initializing values based on which we will be 
class CNNDataLayer(data.Dataset):
    def __init__(self, data_root, filenames, loader):
        self.data_root = data_root
        self.filenames = filenames
        self.loader = loader

    def __getitem__(self, index):
        filename = self.filenames[index]
        target = [int(filename[-5])]
        target = np.array(target)
        target = torch.from_numpy(target)
        data = self.loader(self.data_root, filename)
        #print(np.array([filename]))
        #print(target)
        return data, target, [filename]

    def __len__(self):
        return len(self.filenames)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_sets_test = CNNDataLayer(
        data_root=data_directory,
        filenames=test_files,
        loader=CNNloader
    )

data_loaders_test = data.DataLoader(
        data_sets_test,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )


#Loading the model
vgg16 = models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./vgg16-397923af.pth"))
num_features = vgg16.classifier[6].in_features
#Removing last layer
features = list(vgg16.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 1)])
features.extend([nn.Sigmoid()])
vgg16.classifier = nn.Sequential(*features)

model_loaded = torch.load("./snapshots/epoch-5-trainerror-0.1447084233261339-testerror-0.15577889447236182.pth", map_location=lambda storage, loc: storage)
vgg16.load_state_dict(model_loaded)

#We will be saving the activations on the hard disk
vgg16.eval()

for batch_idx, (data_now, target_now, filenames_current) in enumerate(data_loaders_test):
    print(batch_idx)
    print('Data shape is')
    print(data_now.shape)
    data_now = data_now.to(device)
    target_now = target_now.to(device)
    target_output_model = vgg16(data_now)
    if batch_idx == 0:
        activations_test = target_output_model.detach().numpy()
        filenames_list = np.array(filenames_current).reshape(-1,1)
        target_list = target_now.detach().numpy()
    else:
        activations_test = np.concatenate((activations_test, target_output_model.detach().numpy()), axis = 0)
        filenames_list = np.concatenate((filenames_list, np.array(filenames_current).reshape(-1,1)), axis = 0)
        target_list = np.concatenate((target_list, target_now.detach().numpy()), axis = 0)


import pandas as pd
activation_dataframe = pd.DataFrame(filenames_list, columns = ['Filename'])
activation_dataframe['Activations'] = activations_test
activation_dataframe['Target'] = target_list
#Saving dataframe to file
activation_dataframe.to_csv('activations_test.csv')

import sklearn
from sklearn.metrics import roc_auc_score 
auroc_score = roc_auc_score(y_true = activation_dataframe['Target'], y_score = activation_dataframe['Activations'])
threshold = 0.5

#Confusion matrix
activation_dataframe['Predicted'] = 0
activation_dataframe.loc[activation_dataframe['Activations'] > threshold, 'Predicted'] = 1

from sklearn.metrics import confusion_matrix
confusion_matrix_table = pd.DataFrame(confusion_matrix(y_true = activation_dataframe['Target'], y_pred = activation_dataframe['Predicted'], labels = [0,1]), columns = [0,1])

#Recall
#Recall is TP / (TP + FN), which we want to maximize
#Basically, if 100 people have tuberculosis, how many are we capturing
from sklearn.metrics import recall_score
recall_score(y_true = activation_dataframe['Target'], y_pred = activation_dataframe['Predicted'], labels = [0,1])

#Precision
#TP / (TP + FP)
#Basically, if we preddicted 100 to be tuberculosis, how many were true.
from sklearn.metrics import precision_score
precision_score(y_true = activation_dataframe['Target'], y_pred = activation_dataframe['Predicted'], labels = [0,1])


#Accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_true = activation_dataframe['Target'], y_pred = activation_dataframe['Predicted'])


#Let's draw ROC curve
from sklearn.metrics import roc_curve
xxx = roc_curve(y_true = activation_dataframe['Target'], y_score = activation_dataframe['Activations'])

plt.figure()
plt.title('ROC Curve')
plt.scatter(x = xxx[0], y = xxx[1])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()