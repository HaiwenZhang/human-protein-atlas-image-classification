import time
import copy
import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from data.dataload import get_dateloaders
from models.net import MyModel
from train import train_model
from torch.optim import lr_scheduler


batch_size = 128
valid_size = 0.1
shuffle_dataset = True
random_seed = 42
num_workers = 5
num_epochs = 100
model_name = "resnet50"
num_labels = 28
shuffle = True
feature_extract = True
use_pretrained=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

csv_file = "/home/haiwen/Haiwen/kaggle-datasets/human-protein-atlas-image-classification/train.csv"
image_folder = "/home/haiwen/Haiwen/kaggle-datasets/human-protein-atlas-image-classification/train/"

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

model = MyModel(model_name, num_labels, feature_extract, use_pretrained)
model.to(device)

image_size = model.image_size()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize
])

criterion = torch.nn.MultiLabelSoftMarginLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dataloaders = get_dateloaders(csv_file=csv_file,
                            data_dir=image_folder,
                            train_transform=train_transform,
                            valid_transform=valid_transform,
                            batch_size=batch_size,
                            random_seed=random_seed,
                            valid_size=valid_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

train_model(model=model, 
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer, 
            scheduler=exp_lr_scheduler,
            device=device,
            num_epochs=num_epochs)


