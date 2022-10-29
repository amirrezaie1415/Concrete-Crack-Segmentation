#!/usr/bin/env python

"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

To train the model, this script must be run in the command prompt. The command-line parsing module "argparse" is used
to take a number of inputs from the user such as the architecture type of the deep learning model, hyper-parameters,
path to the data and etc. You may find all arguments in the "if block" at the last part of the script.
"""

# import necessary modules
import argparse
import glob
import os
from solver import Solver
from data_loader import CrackDatasetTrainVal
from torch.backends import cudnn
import random
import torch
import numpy as np
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# To have reproducible results the random seed is set to 42.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(config):
    if config.model_type not in ['TernausNet16']:
        print(
            'ERROR!! model_type should be selected in TernausNet16')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories for results if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.logs_path):
        os.makedirs(config.logs_path)

    print(config)
    image_suffix = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    # Load training data
    resizing_factor = 0.5
    train_transform = A.Compose(
        [A.RandomRotate90(p=config.augmentation_prob),
         A.GridDistortion(p=config.augmentation_prob),
         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=config.augmentation_prob),
         A.Resize(int(config.image_size * resizing_factor), int(config.image_size * resizing_factor),
                  p=config.augmentation_prob),
         A.PadIfNeeded(min_height=config.image_size, min_width=config.image_size, border_mode=0, value=0),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()
         ]
    )
    train_images_filenames = [i for i in os.listdir(config.train_path) if i.lower().endswith(image_suffix)]
    train_dataset = CrackDatasetTrainVal(images_filenames=train_images_filenames,
                                         images_directory=config.train_path,
                                         masks_directory=config.train_annotatoin_path,
                                         transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True)

    print("image count in {} path :{}".format('number of training data:', len(train_loader.dataset)))

    # Load validation data
    valid_transform = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )
    valid_images_filenames = [i for i in os.listdir(config.valid_path) if i.lower().endswith(image_suffix)]
    valid_dataset = CrackDatasetTrainVal(images_filenames=valid_images_filenames,
                                         images_directory=config.valid_path,
                                         masks_directory=config.valid_annotatoin_path,
                                         transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_workers,
                              pin_memory=True)
    print("image count in {} path :{}".format('number of validation data:', len(valid_loader.dataset)))

    # Load test data
    if config.test_path != "":
        test_images_filenames = [i for i in os.listdir(config.test_path) if i.lower().endswith(image_suffix)]
        test_dataset = CrackDatasetTrainVal(images_filenames=test_images_filenames,
                                            images_directory=config.test_path,
                                            masks_directory=config.test_annotatoin_path,
                                            transform=valid_transform)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True)
        print("image count in {} path :{}".format('number of test data:', len(test_loader.dataset)))
    else:
        test_loader = None

    # Define a solver instance
    solver = Solver(config, train_loader, valid_loader, test_loader)
    solver.train()
    if config.test_path != "":
        solver.pred()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)  # input size of the model (width = height)
    parser.add_argument('--img_ch', type=int, default=1, help='number of channels of the input data')
    parser.add_argument('--output_ch', type=int, default=1, help='number of channels of the output data')
    parser.add_argument('--pretrained', type=int, default=1, help='to use pre-trained weights input must be 1')
    parser.add_argument('--num_epochs', type=int, default=100, help='number epochs for training')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help=' momentum1 in the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum2 in the Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lossfunc', type=str, default='DiceLoss', help='DiceLoss')
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--number_layers_freeze', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='TernausNet16', help='TernausNet16')

    parser.add_argument('--model_path', type=str, default='../models', help='path to save the best model')
    parser.add_argument('--train_path', type=str, default='../dataset/train/', help='path to training images')
    parser.add_argument('--train_annotatoin_path', type=str, default='../dataset/train_GT/',
                        help='path to training annotation images')

    parser.add_argument('--valid_path', type=str, default='../dataset/valid/', help='path to validation images')
    parser.add_argument('--valid_annotatoin_path', type=str, default='../dataset/valid_GT/',
                        help='path to validation annotation images')

    parser.add_argument('--test_path', type=str, default='', help='path to test images')
    parser.add_argument('--test_annotatoin_path', type=str, default='',
                        help='path to test annotation images')

    parser.add_argument('--logs_path', type=str, default='../logs/', help='path to save results')
    parser.add_argument('--cuda_idx', type=int, default=1, help='if cuda available = 1 otherwise = 0')
    config = parser.parse_args()
    main(config)
