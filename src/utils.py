import numpy as np


def zero_pad(image, desired_size):

    remainder_y = image.shape[0] % desired_size
    if remainder_y != 0:
        newImgSize_y = image.shape[0] + desired_size - remainder_y
    else:
        newImgSize_y = image.shape[0]
    remainder_x = image.shape[1] % desired_size
    if remainder_x != 0:
        newImgSize_x = image.shape[1] + desired_size - remainder_x
    else:
        newImgSize_x = image.shape[1]

    if image.ndim == 3:
        new_im = np.zeros((newImgSize_y, newImgSize_x, 3), dtype="uint8")
        new_im[0:image.shape[0], 0:image.shape[1], :] = image[:, :, :]
        return new_im
    if image.ndim == 2:
        new_im = np.zeros((newImgSize_y, newImgSize_x), dtype="uint8")
        new_im[0:image.shape[0], 0:image.shape[1]] = image[:, :]
        return new_im


def sliding_window(image, step_size, window_size):
    if image.ndim == 3:
        # slide a window across the image
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                # yield the current window
                yield x, y, image[y:y + window_size[1], x:x + window_size[0], :]
    if image.ndim == 2:
        # slide a window across the image
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                # yield the current window
                yield x, y, image[y:y + window_size[1], x:x + window_size[0]]
