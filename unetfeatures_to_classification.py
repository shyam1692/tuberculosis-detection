# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:41:46 2018

@author: Shyam
MLSP Tuberculosis data exploration
"""

import os
os.chdir('C:\stuff\Studies\Fall 18\Machine Learning in Signal Processing\Project\Tuberculosis Detection\Code')


import numpy as np
#Checking unique image sizes of all images, and seeing if we need to resize or not.

#os.listdir(data_directory)
data_directory = '../Data/ChinaSet_AllFiles/CXR_png'

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


#What all data transformations can we do
#What normalization
#How to initialize weights
#Save model at each epoch and check the graph of training error and test error by iterations. It should decrease.
#Check AUROC (balanced dataset, so not too much of problem)
#Keep the code running by tmux
#Use CUDA, GPU, with number of workers as minimum 4

#
"""
First task, we will split data into training and validation
Since our first step is to run VGG16, we will be resizing to the input size for the VGG net
224*224

Also, VGG16 is trained on RGB / 3 channel image
So, we can make it into 3 channel

Initially, let's count the number of channels and get an idea.
"""
#from scipy import misc

all_files = os.listdir(data_directory)
del(all_files[len(all_files) - 1])



#shapes = {}
#count_multichannel = 0
#multichannel_file = []
#singlechannel_files = []
#for filename in  all_files:
#    filename_actual = data_directory + '/' +filename
#    #print(filename_actual)
#    image_trial = io.imread(filename_actual)
#    if len(image_trial.shape) == 2:
#        singlechannel_files.append(filename)
#    if len(image_trial.shape) > 2:
#        if image_trial.shape[2] == 3:
#            count_multichannel += 1
#            multichannel_file.append(filename)
#    if image_trial.shape in shapes.keys():
#        shapes[image_trial.shape] += 1
#    else:
#        shapes[image_trial.shape] = 1
#        
#len(shapes.keys())

"""
In China Dataset, all images are 3 channel images
In Montgomery Set, all images are 2 channel images.

As of now, let's train using chinese dataset.

Divide Chinese dataset into Train and validation, and resize to 224 * 224
"""

data_directory = '../Data/ChinaSet_AllFiles/CXR_png'
#Loading from all_files in data_directory
#Data loader we will define as batch size 8 and number of workers as 4
all_files = np.array(all_files)
#singlechannel_files = np.array(singlechannel_files)
np.random.shuffle(all_files)

train_files = all_files[:round(0.7*all_files.shape[0])]
test_files = all_files[round(0.7*all_files.shape[0]):]

"""
Let's see if the classes are balanced, else we would have to reshuffle
"""
categories = {'0':0,'1':0}
for file in test_files:
    categories[file[-5]] += 1

"""
It's balanced

We now create a dataloader function, which would be resizing the data and maybe normalizing by dividing by 255
and also have output
"""
#The below transformation converts the image into torch readable format, into channels*height*width, and 
#Does the normalization of dividing by 255
#data_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)), transforms.ToTensor()])

#This one slightly changed
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
        return data, target

    def __len__(self):
        return len(self.filenames)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_sets_train = CNNDataLayer(
        data_root=data_directory,
        filenames=train_files,
        loader=CNNloader
    )

data_sets_test = CNNDataLayer(
        data_root=data_directory,
        filenames=test_files,
        loader=CNNloader
    )
#data_sets[0]
#data_sets.data_root
#filename = singlechannel_files[0]
#CNNloader(data_directory, singlechannel_files[0])

"""
In case of GPU, we will set the num_workers to be more than 1, like 4
to_device will also be set to CUDA
"""
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





#Abhialshs model
"""
Load the model here
"""
from unet_model import unet
model_unet = unet()
model_unet = model_unet.to(device)

#Freezing before last layer.
for param in model_unet.features.parameters():
    param.require_grad = False

from classifier_on_unet import ClassifierOnUnet
model_to_train = ClassifierOnUnet()
model_to_train = model_to_train.to(device)
#Setting loss criteria and optimizer here
criterion = nn.BCELoss()
optimizer_ft = optim.SGD(model_to_train.parameters(), lr=0.001, momentum=0.9)
num_epochs = 100
"""
Now since we have defined everything above, we will start training here
"""
for epoch in range(1, num_epochs+1):
    print('Epoch is ' + str(epoch))
    train_loss = 0
    test_loss = 0
    total_misclassified_train = 0
    total_count_train = 0
    total_misclassified_test = 0
    total_count_test = 0
    
    with torch.set_grad_enabled(True):
        for batch_idx, (data_now, target_now) in enumerate(data_loaders_train):
            model_to_train.train()
            model_unet.eval()
            print(batch_idx)
            print('Data shape is')
            print(data_now.shape)
    #        print('Target shape is')
    #        print(target_now.shape)    
            data_now = data_now.to(device)
            target_now = target_now.to(device)
            features_output_model = model_unet.features(data_now)
            target_output_model = model_to_train(features_output_model)
            target_now = target_now.type(torch.FloatTensor)
            target_now = target_now.to(device)
            #target_now = Variable(target_now)
            target_loss = criterion(target_output_model, target_now)
            #Estimating accuracy
            misclassified_temp = target_output_model[target_now == 1]
            misclassified_count_positive = misclassified_temp[misclassified_temp < 0.5].size()[0]
            misclassified_temp = target_output_model[target_now == 0]
            misclassified_count_negative = misclassified_temp[misclassified_temp >= 0.5].size()[0]
            total_misclassified_train += misclassified_count_positive + misclassified_count_negative
            total_count_train += data_now.shape[0]
            
            optimizer_ft.zero_grad()
            target_loss.backward()
            optimizer_ft.step()
    
    with torch.set_grad_enabled(False):
        for batch_idx, (data_now, target_now) in enumerate(data_loaders_test):
            model_to_train.eval()
            model_unet.eval()
            print(batch_idx)
            print('Data shape is')
            print(data_now.shape)
            data_now = data_now.to(device)
            target_now = target_now.to(device)
            features_output_model = model_unet.features(data_now)
            target_output_model = model_to_train(features_output_model)

            misclassified_temp = target_output_model[target_now == 1]
            misclassified_count_positive = misclassified_temp[misclassified_temp < 0.5].size()[0]
            misclassified_temp = target_output_model[target_now == 0]
            misclassified_count_negative = misclassified_temp[misclassified_temp >= 0.5].size()[0]
            total_misclassified_test += misclassified_count_positive + misclassified_count_negative
            total_count_test += data_now.shape[0]
        
    #Saving the weights now
    snapshot_path = './snapshots_trial'
    if not os.path.isdir(snapshot_path):
        os.makedirs(snapshot_path)
    snapshot_name = 'epoch-{}-trainerror-{}-testerror-{}.pth'.format(epoch, 
                           float(total_misclassified_train / total_count_train),
                           float(total_misclassified_test / total_count_test))
    
    torch.save(model_to_train.state_dict(), os.path.join(snapshot_path, snapshot_name))
    
    
#After saving the model
#Note, you have to create the model using the same structure as defined
model_loaded = torch.load("./snapshots/epoch-5-trainerror-0.1447084233261339-testerror-0.15577889447236182.pth", map_location=lambda storage, loc: storage)
model_to_train.load_state_dict(model_loaded)

