# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import ndimage, signal
from os import listdir
from os.path import isfile, join
from PIL import Image

HEIGHT = 480
WIDTH = 616

debug = False


def read_img(path):
    # we load the image in grayscale
    im = Image.open(path)
    im = np.asarray(im)
    im = np.float32(im)

    maxI = np.max(im)
    minI = np.min(im)
    im = (im - minI) / (maxI - minI)
    im = np.asarray(im)

    """if(debug):
        cv2.imshow('img before norm',img)
        cv2.waitKey()"""
    if debug:
        cv2.imshow("img after norm", img)
        cv2.waitKey()

    """img_3d = np.zeros((480,616,3),dtype= np.uint8)
    img_3d[:,:,0]=img*255
    img_3d[:,:,1]=img*255
    img_3d[:,:,2]=img*255
    cv2.imshow('img with 3 channels', img_3d)
    cv2.waitKey()"""

    # Pytorch expects us to give an array of size [batch_size, channels, height, width]
    # So we need to expand our array to create the one channel of the image
    im = im[np.newaxis, :, :]
    assert np.shape(im) == (1, HEIGHT, WIDTH), "Incorrect shape of image"
    return im


def normalize_img(img):
    # might have to check if image is plane
    return (img - img.min()) / (img.max() - img.min())
    # return (img  / (img.max()))*255


def read_landmarks(path):
    # read in landmarks as an array of size 23x2 (x,y)
    file = open(path, "r")
    positions = [
        [int(point.split(";")[0]), int(point.split(";")[1])]
        for point in file.readlines()
    ]
    return positions


def gen_landmark_channels(positions):
    # get landmarks and creates channels(one channel per landmark) with a gaussian around the landmark
    channels = np.zeros((23, HEIGHT, WIDTH))
    for idx, point in enumerate(positions):
        # check if point is inside boundaries
        # first number is the x (width) position, the second one is the y (height) position
        if point[0] >= 0 and point[0] < WIDTH and point[1] >= 0 and point[1] < HEIGHT:
            # we need to save the landmark of channel idx as #channel ,height, width (idx,y,x)
            channels[idx, point[1], point[0]] = 1
            # generates blob around landmark
            channels[idx, :, :] = ndimage.filters.gaussian_filter(
                channels[idx, :, :], 40
            )
            channels[idx, :, :] = normalize_img(channels[idx, :, :])

    return channels


def get_folders(path):
    onlyfolders = [f for f in listdir(path) if not isfile(join(path, f))]
    return onlyfolders


def get_files(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles


def gkern_scipy(size, sigma):
    unit_impulse = signal.unit_impulse(size, "mid")
    return ndimage.filters.gaussian_filter(unit_impulse, sigma).reshape(30, 30)


def read_img_matthias(path):
    im = Image.open(path)
    im = np.asarray(im)
    im = np.float32(im)

    maxI = np.max(im)
    minI = np.min(im)
    im = (im - minI) / (maxI - minI)
    im = np.asarray(im)

    im = im * 255
    im = im.astype(np.uint8)
    return im


if __name__ == "__main__":
    # read_img('data/X_Ray_Data/train/_z_137.588_x_-30_y_-30_roll_20_yaw_15_0/_0__y_-60.0_r_-60.0_y_-30.0_x_-30.0_z_181.6_xray.png')
    # read_img('res2.png')
    img = read_img_matthias("test2.png")
    positions = read_landmarks("test2.txt")
    channels = gen_landmark_channels(positions)
    print(channels.shape)

    if debug:
        for i in range(channels.shape[0]):
            img = np.maximum(img, channels[i, :, :] * 255)
            blob = channels[i, :, :] * 255
            blob = blob.astype(np.uint8)
            cv2.imwrite("blob" + str(i) + ".png", blob)

    cv2.imwrite("img_blobs.png", img)
