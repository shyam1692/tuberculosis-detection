# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:04:15 2018

@author: Shyam

Let's try Unet Model right now


"""
import os
os.chdir('C:\stuff\Studies\Fall 18\Machine Learning in Signal Processing\Project\Tuberculosis Detection\Code')


import numpy as np
#Checking unique image sizes of all images, and seeing if we need to resize or not.

#os.listdir(data_directory)
data_directory = '../Data/ChinaSet_AllFiles/CXR_png'
mask_data_directory = '../Data/ChinaSet_AllFiles/mask'

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

"""Defining end"""

all_files = os.listdir(mask_data_directory)

#Checking balance
categories = {'0':0,'1':0}
for file in all_files:
    categories[file[-10]] += 1
    
"""Its balanced"""
all_files = np.array(all_files)
np.random.shuffle(all_files)

train_files = all_files[:round(0.7*all_files.shape[0])]
test_files = all_files[round(0.7*all_files.shape[0]):]


categories = {'0':0,'1':0}
for file in test_files:
    categories[file[-10]] += 1


data_transforms = transforms.Compose([ transforms.ToTensor()])

def CNNloader(data_root, filename):
    filename_actual = data_root + '/' + filename
    data_old = io.imread(filename_actual)
    data_old = resize(data_old,(256,256))
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

#import cv2
class CNNDataLayer(data.Dataset):
    def __init__(self, data_root, filenames, loader, mask_root):
        self.data_root = data_root
        self.filenames = filenames
        self.loader = loader
        self.mask_root = mask_root

    def __getitem__(self, index):
        filename = self.filenames[index]
        filename_input = filename[:-9] + '.png'
        filename_mask = self.mask_root + '/' + filename
        target = io.imread(filename_mask)
        if len(target.shape) > 2:
            target = np.array(target[:,:,0])
        target = np.array(target)
        target = resize(target,(256,256), preserve_range = True)
        target[target < 200] = 0
        target[target >= 200] = 1
        target = target.reshape((1,) + target.shape)
        target = torch.from_numpy(target)
        data = self.loader(self.data_root, filename_input)
        return data, target

    def __len__(self):
        return len(self.filenames)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_sets_train = CNNDataLayer(
        data_root=data_directory,
        filenames=train_files,
        loader=CNNloader,
        mask_root = mask_data_directory
    )

data_sets_test = CNNDataLayer(
        data_root=data_directory,
        filenames=test_files,
        loader=CNNloader,
        mask_root = mask_data_directory
    )



data_loaders_train = data.DataLoader(
        data_sets_train,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

data_loaders_test = data.DataLoader(
        data_sets_test,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

from unet_model import unet
model_to_train = unet()
model_to_train = model_to_train.to(device)

criterion = nn.BCELoss()
optimizer_ft = optim.SGD(model_to_train.parameters(), lr=0.000001, momentum=0.9)
num_epochs = 100


total_loss_examples = {}
total_loss_examples['train'] = 0
total_loss_examples['test'] = 0


"""Lets train model now"""
for epoch in range(1, num_epochs+1):
    print('Epoch is ' + str(epoch))
    train_loss = 0
    test_loss = 0
    total_misclassified_train = 0
    total_count_train = 0
    total_misclassified_test = 0
    total_count_test = 0
    print('Training started')
    true_positives_train = 0
    false_negatives_train = 0
    true_negatives_train = 0
    false_positives_train = 0
    with torch.set_grad_enabled(True):
        for batch_idx, (data_now, target_now) in enumerate(data_loaders_train):
            model_to_train.train()
            print(batch_idx)
            print('Data shape is')
            print(data_now.shape)
    #        print('Target shape is')
    #        print(target_now.shape)    
            data_now = data_now.to(device)
            target_now = target_now.to(device)
            target_output_model = model_to_train(data_now)
            target_now = target_now.type(torch.FloatTensor)
            target_now = target_now.to(device)
            #target_now = Variable(target_now)
            target_loss = criterion(target_output_model, target_now)
            all_positives = target_output_model[target_now == 1]
            true_positives = all_positives[all_positives > 0.5].size()[0]
            false_negatives = all_positives.size()[0] - true_positives
            
            all_negatives = target_output_model[target_now == 0]
            true_negatives = all_negatives[all_negatives <= 0.5].size()[0]
            false_positives = all_negatives.size()[0] - true_negatives
            
            
            true_positives_train += true_positives
            false_negatives_train += false_negatives
            true_negatives_train += true_negatives
            false_positives_train += false_positives
            print('True positives train so far')
            print(true_positives_train)
            total_loss_examples['train'] += target_loss.item()*data_now.shape[0]
            #Estimating accuracy
#            misclassified_temp = target_output_model[target_now == 1]
#            misclassified_count_positive = misclassified_temp[misclassified_temp < 0.5].size()[0]
#            misclassified_temp = target_output_model[target_now == 0]
#            misclassified_count_negative = misclassified_temp[misclassified_temp >= 0.5].size()[0]
#            total_misclassified_train += misclassified_count_positive + misclassified_count_negative
#            total_count_train += data_now.shape[0]
#            
            optimizer_ft.zero_grad()
            target_loss.backward()
            optimizer_ft.step()
    
    print('Testing has started')
    true_positives_test = 0
    false_negatives_test = 0
    true_negatives_test = 0
    false_positives_test = 0
    with torch.set_grad_enabled(False):
        for batch_idx, (data_now, target_now) in enumerate(data_loaders_test):
            model_to_train.eval()
            print(batch_idx)
            print('Data shape is')
            print(data_now.shape)
            data_now = data_now.to(device)
            target_now = target_now.to(device)
            target_now = target_now.type(torch.FloatTensor)
            target_now = target_now.to(device)
            target_output_model = model_to_train(data_now)
            target_loss = criterion(target_output_model, target_now)
            all_positives = target_output_model[target_now == 1]
            true_positives = all_positives[all_positives > 0.5].size()[0]
            false_negatives = all_positives.size()[0] - true_positives
            
            all_negatives = target_output_model[target_now == 0]
            true_negatives = all_negatives[all_negatives <= 0.5].size()[0]
            false_positives = all_negatives.size()[0] - true_negatives
            
            
            true_positives_test += true_positives
            false_negatives_test += false_negatives
            true_negatives_test += true_negatives
            false_positives_test += false_positives
            print('True positives test so far')
            print(true_positives_test)
            total_loss_examples['test'] += target_loss.item()*data_now.shape[0]
#            misclassified_temp = target_output_model[target_now == 1]
#            misclassified_count_positive = misclassified_temp[misclassified_temp < 0.5].size()[0]
#            misclassified_temp = target_output_model[target_now == 0]
#            misclassified_count_negative = misclassified_temp[misclassified_temp >= 0.5].size()[0]
#            total_misclassified_test += misclassified_count_positive + misclassified_count_negative
#            total_count_test += data_now.shape[0]
    #Calculating precision and recall for test and train
    precision_train = (true_positives_train) / (true_positives_train + false_positives_train)    
    precision_test = (true_positives_test) / (true_positives_test + false_positives_test)
    recall_train = (true_positives_train) / (true_positives_train + false_negatives_train)
    recall_test = (true_positives_test) / (true_positives_test + false_negatives_test)
    #calculating f1 score train and test
    f1_train = 2*(precision_train*recall_train) / (precision_train + recall_train)    
    f1_test = 2*(precision_test*recall_test) / (precision_test + recall_test)
    #Saving the weights now
    snapshot_path = './snapshots_trial_unet'
    if not os.path.isdir(snapshot_path):
        os.makedirs(snapshot_path)
    snapshot_name = 'epoch-{}-trainF1-{}-testF1-{}-trainloss-{}-testloss-{}.pth'.format(epoch, f1_train, f1_test,
                           float(float(total_loss_examples['train']) / float(len(data_loaders_train.dataset))),
                           float(float(total_loss_examples['test']) / float(len(data_loaders_test.dataset)))
                           )
    
    torch.save(model_to_train.state_dict(), os.path.join(snapshot_path, snapshot_name))
