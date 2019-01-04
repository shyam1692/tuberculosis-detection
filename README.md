# Tuberculosis Detection by Deep Learning
Contributors: Abhilash Kuhikar, Karen Sanchez, Shyam Narasimhan
Please see Final Report for implementation details

## Goal:
To build a platform for automatically diagnosing a patient with tuberculosis and detecting the relevant symptoms from CXRs using Machine Learning techniques.
We used Grad-cam for visual explanation from Deep Networks to identify the regions of Chest X-ray where the network concentrates for learning the classification.

The source code for grad-cam was used from here: https://github.com/jacobgil/pytorch-grad-cam

We attempt to make the network to learn from CXR from a radiologist's perspective

## Highlights of Methodologies:
1. Classification using VGGNet
2. Classification using Own CNN
3. Deriving latent features from a network designed for Semantic Segmentation of lungs and using them for classification

## Future Work
1. Annotating the chest regions for abnormalities and detecting them
2. Generating clinical readings from the CXRs
3. Building Visualization pipeline more robust to different changes in network architectures
