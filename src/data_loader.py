"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

The functions and classes defined in this script are used to load the images and ground truth files (as batches)
and apply randomly some transformations as a means of data augmentation.
"""

# import necessary modules
import os
from torch.utils import data
import numpy as np
import cv2


class CrackDatasetTrainVal(data.Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_directory, image_filename.replace(".png", "_mask.png")),
                          cv2.IMREAD_GRAYSCALE)
        mask = (mask / mask.max()).astype(np.float64)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)
        return image, mask


class CrackDatasetInference(data.Dataset):
    def __init__(self, images_filenames, images_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image
