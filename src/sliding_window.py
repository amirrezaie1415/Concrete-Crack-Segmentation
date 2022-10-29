# -*- coding: utf-8 -*-


## Load necessary packages and instances
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
import cv2


def zero_pad(image, desired_size): # imgae : RGB

    remainder_y = image.shape[0] % desired_size
    newImgSize_y = image.shape[0] + desired_size - remainder_y
    remainder_x = image.shape[1] % desired_size
    newImgSize_x = image.shape[1] + desired_size - remainder_x

    if image.ndim == 3:
        new_im = np.zeros((newImgSize_y, newImgSize_x, 3), dtype=str(image.dtype))
        new_im[0:image.shape[0], 0:image.shape[1], :] = image[:,:,:]
        return new_im
    if image.ndim == 2:
        new_im = np.zeros((newImgSize_y, newImgSize_x), dtype=str(image.dtype))
        new_im[0:image.shape[0], 0:image.shape[1]] = image[:,:]
        return new_im 

def sliding_window(image, stepSize, windowSize):
    if image.ndim == 3:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0], :])
    if image.ndim == 2:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
                
