# #Includes all code related to the dataset,
# # such as the dataset class, preprocessing,
# # augmentation, and post-processing routines.
#

import os
import torch
from torchvision import datasets, transforms

def data_transforms(phase):
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform

def load_datasets_and_dataloaders(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms(x)) for x in ['train', 'val', 'test']}
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=1, shuffle=True)
    }

    return dataloaders

