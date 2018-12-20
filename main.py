import os
import argparse
import logging
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from data.dataload import get_dateloaders
from models.net import MyModel
from train import train_and_evaluate
from torch.optim import lr_scheduler

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help="Direcotry containning the image data")
parser.add_argument("--model_dir", help="Directory containning params.json")

shuffle_dataset = True
shuffle = True
feature_extract = True
use_pretrained = True

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def setup_and_train(parmas):
    model = MyModel(params).cuda() if params.cuda else MyModel(params)

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

    loss_fn = torch.nn.MultiLabelSoftMarginLoss()

    # Observe that all parameters are being optimized
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=params.step_size, gamma=params.gama)

    dataloaders = get_dateloaders(params,
                                  train_transform=train_transform,
                                  valid_transform=valid_transform)

    train_and_evaluate(model=model,
                       dataloaders=dataloaders,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       scheduler=exp_lr_scheduler,
                       params=params)


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    params.csv_file = os.path.join(args.image_dir, 'train.csv')
    params.data_dir = os.path.join(args.image_dir, 'train')
    params.model_metrics_file = os.path.join(args.model_dir, "metrics.csv")
    params.shuffle_dataset = shuffle_dataset
    params.shuffle = shuffle
    params.feature_extract = feature_extract
    params.use_pretrained = use_pretrained

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    setup_and_train(params)
    
    # save train metrics data to model csv file
    utils.save_train_metrics(params)
