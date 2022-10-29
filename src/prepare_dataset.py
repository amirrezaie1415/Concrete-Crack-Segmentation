import glob

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io
from sliding_window import sliding_window, zero_pad
from skimage.morphology import skeletonize


# verificaiton
path_rgb = '../../datasets/Concrete Crack Segmentation Dataset/original_dataset/rgb/'
path_bw = '../../datasets/Concrete Crack Segmentation Dataset/original_dataset/BW/'


imgs = glob.glob(os.path.join(path_rgb, '*.jpg'))
bws = glob.glob(os.path.join(path_bw, '*.jpg'))


imgs_names = set([img.split(os.sep)[-1] for img in imgs])
bws_names = set([bw.split(os.sep)[-1] for bw in bws])
intersection_names = imgs_names.intersection(bws_names)
intersection_names = [*intersection_names]


### sliding window

img_patches_path = '../../datasets/Concrete Crack Segmentation Dataset/treated_dataset/images/'
mask_patches_path = '../../datasets/Concrete Crack Segmentation Dataset/treated_dataset/masks/'
desired_size = 256
#
for ind in range(len(intersection_names)):
    img_stem = os.path.basename(intersection_names[ind]).split('.')[0]
    img = skimage.io.imread(os.path.join(path_rgb, intersection_names[ind]))
    mask = skimage.io.imread(os.path.join(path_bw, intersection_names[ind]))

    padded_img = zero_pad(img, desired_size=desired_size)
    padded_mask = zero_pad(mask, desired_size=desired_size)

    img_windows = []
    xys = []
    for (x, y, window) in sliding_window(padded_img, desired_size, windowSize=(desired_size, desired_size)):
        img_windows.append(window)
        xys.append((x, y))
    mask_windows = []
    for (x, y, window) in sliding_window(padded_mask, desired_size, windowSize=(desired_size, desired_size)):
        mask_windows.append(window)

    for mask_window, img_window, xy in zip(mask_windows, img_windows, xys):
        if len(np.where(mask_window == 255)[0]) > 50:
            save_dir = os.path.join(img_patches_path,
                                    img_stem + "_{:d}".format(xy[0]) + "_{:d}".format(xy[1]) + '.png')
            skimage.io.imsave(save_dir, img_window)
            save_dir = os.path.join(mask_patches_path,
                                    img_stem + "_{:d}".format(xy[0]) + "_{:d}".format(xy[1]) + '_mask.png')
            skimage.io.imsave(save_dir, mask_window)



### skletonize masks

# core = 'Core3'
# mask_patches_fullannot_path = '../../dataset/treated/' + core + '/masks/patches_full_annot_256'
# mask_patches_skleton_path = '../../dataset/treated/' + core + '/masks/patches_skleton_256'
#
# masks_list = glob.glob(os.path.join(mask_patches_fullannot_path, '*.png'))
# masks_list.sort()
#
# for ind in range(len(masks_list)):
#     mask = skimage.io.imread(masks_list[ind])
#     skleton = skeletonize(mask/255)
#     skleton = skleton.astype(np.uint8) * 255
#
#     img_name = os.path.basename(masks_list[ind])
#     save_dir = os.path.join(mask_patches_skleton_path, img_name)
#     skimage.io.imsave(save_dir, skleton)


## create dataset
import shutil

root = '/home/swissinspect/Projects/malek_crack_detection/Stone-crack-segmentation/'
train_images = os.path.join(root, 'dataset', 'train')
train_masks = os.path.join(root, 'dataset', 'train_GT')
os.makedirs(train_images, exist_ok=True)
os.makedirs(train_masks, exist_ok=True)

valid_images = os.path.join(root, 'dataset', 'valid')
valid_masks = os.path.join(root, 'dataset', 'valid_GT')
os.makedirs(valid_images, exist_ok=True)
os.makedirs(valid_masks, exist_ok=True)

test_images = os.path.join(root, 'dataset', 'test')
test_masks = os.path.join(root, 'dataset', 'test_GT')
os.makedirs(test_images, exist_ok=True)
os.makedirs(test_masks, exist_ok=True)


path_to_cores = '/home/swissinspect/Projects/malek_crack_detection/dataset/treated'

train_cores = ['Core0', 'Core3']
valid_cores = ['Core2']
test_cores = ['Core1']

for core in train_cores:
    imgs = glob.glob(os.path.join(path_to_cores, core, 'images', 'patches_256', '*.png'))
    for img in imgs:
        dst = os.path.join(train_images, os.path.basename(img))
        shutil.copyfile(img, dst)

    masks = glob.glob(os.path.join(path_to_cores, core, 'masks', 'patches_full_annot_256', '*.png'))
    for mask in masks:
        dst = os.path.join(train_masks, os.path.basename(mask))
        shutil.copyfile(mask, dst)

for core in valid_cores:
    imgs = glob.glob(os.path.join(path_to_cores, core, 'images', 'patches_256', '*.png'))
    for img in imgs:
        dst = os.path.join(valid_images, os.path.basename(img))
        shutil.copyfile(img, dst)

    masks = glob.glob(os.path.join(path_to_cores, core, 'masks', 'patches_full_annot_256', '*.png'))
    for mask in masks:
        dst = os.path.join(valid_masks, os.path.basename(mask))
        shutil.copyfile(mask, dst)

for core in test_cores:
    imgs = glob.glob(os.path.join(path_to_cores, core, 'images', 'patches_256', '*.png'))
    for img in imgs:
        dst = os.path.join(test_images, os.path.basename(img))
        shutil.copyfile(img, dst)

    masks = glob.glob(os.path.join(path_to_cores, core, 'masks', 'patches_full_annot_256', '*.png'))
    for mask in masks:
        dst = os.path.join(test_masks, os.path.basename(mask))
        shutil.copyfile(mask, dst)



