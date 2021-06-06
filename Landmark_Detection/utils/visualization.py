# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torchvision as tv
import torch


def compare_prediction_target(img, prediction, target):
    # fucntion that shows both images with blobs on it
    prediction_img = transform_img_into_3channels(img)
    target_img = transform_img_into_3channels(img)
    for i in range(prediction.shape[0]):
        prediction_img = np.maximum(prediction_img[:, :, 0], prediction[i, :, :] * 255)
        target_img = np.maximum(target_img[:, :, 1], target[i, :, :] * 255)
    plot_prediction_target(prediction_img, target_img)


def plot_prediction_target(prediction_img, target_img):
    window = np.zeros((960, 616, 3), dtype=np.uint8)
    window[:480, :, :] = prediction_img
    window[480:, :, :] = target_img
    cv2.imwrite("Plottet_prediction_target", window)


def add_blob_number_2_image(img, number, position, is_prediction):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # position = tuple (x,y)
    color = (255, 0, 0) if is_prediction else (0, 255, 0)
    cv2.putText(img, str(number), position, font, 1, color, 2, cv2.LINE_AA)


def convert_grayscale_to_rgb(images):
    # takes a whole batch and extends dimension 1 (channels)
    imgs_rgb = torch.zeros(images.size(0), 3, 480, 616)
    for idx, img in enumerate(images):
        img_rgb = torch.cat((img, img, img), 0)
        imgs_rgb[idx] = img_rgb
    return imgs_rgb


def compare_output_target(data_imgs, outputs, targets):
    imgs_rgb = convert_grayscale_to_rgb(data_imgs)
    print(torch.max(imgs_rgb))
    print(torch.max(outputs))
    print(outputs.size())
    print(torch.max(targets))
    for sample_idx in range(outputs.shape[0]):
        for channel_idx in range(outputs[sample_idx].shape[0]):
            imgs_rgb[sample_idx, 0] = torch.max(
                imgs_rgb[sample_idx, 0], targets[sample_idx, channel_idx]
            )
            imgs_rgb[sample_idx, 1] = torch.max(
                imgs_rgb[sample_idx, 1], outputs[sample_idx, channel_idx]
            )

    # expand img to 3 channels
    # mark landmarks of output in red, target in green
    # add number of landmark pos
    # return image
    return imgs_rgb
