## Cervical Cancer Classification Using Deep Learning

Course project for Deep Learning for Medical Imaging, Johns Hopkins University (Fall 2025)

## Overview
Image-based cervical cancer classification comparing EfficientNet-B0 (transfer learning) 
with a baseline CNN on VIA screening images from IARC and Jhpiego.

## Methods
- EfficientNet-B0 with ImageNet pre-trained weights
- Class-weighted cross-entropy loss for imbalanced data
- Data augmentation (random crop, rotation, color jitter)
- Cosine annealing learning rate scheduler
- Early stopping based on validation F1

## Results
- EfficientNet-B0: 63% accuracy, 0.56 macro-F1
- Baseline CNN: 39% accuracy, 0.33 macro-F1

## Tools
Python, PyTorch, scikit-learn, matplotlib
