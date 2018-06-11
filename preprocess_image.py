from skimage.io import imread, imshow, imsave
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re


def normalize(x):
    '''?????

    Args:
        x: ????

    Returns:
        Array of normalized data

    '''
    return Normalizer(x, norm='l2')


def one_hot_encode(labels):
    '''??????????

    Args:
        labels: ??

    Returns:
        Array of one-hot-encoded data

    '''
    return LabelBinarizer().fit_transform(labels)


def get_filename_from_path(file_path):
    '''???????????

    Args:
        file_path: ????(??????????)

    Return:
        ???, ???

    Raise:
        RuntimeError: File name not found
        RuntimeError: Directory name not found
    '''
    filename = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)
    if not filename:
        raise RuntimeError('File name not found')

    if not dirname:
        raise RuntimeError('Directory name not found')
    return filename, dirname


def image_load(image_path, show_image=False):
    '''????????

    Args:
        image_path: ????
        show_image: ???????

    Return:
        Array of image
    '''
    img_load = cv2.imread(image_path)
    if show_image:
        cv2.destroyAllWindows()
        plt.imshow(img_load)
        plt.show()
    return img_load


def resize_image(resize_image_directory, image_path, resize, show_image=False):
    '''??????

    ?????????????????????

    Args:
        resize_image_directory: ????????????
        image_path: ????/????
        resize: ??????
        show_image: ???????

    Return:
        Array of image
    '''
    file_name, dirctory_name = get_filename_from_path(image_path)
    resize_img_path = resize_image_directory + '/' + file_name
    if not os.path.exists(resize_image_directory):
        os.makedirs(resize_image_directory)

    img_load = image_load(image_path)
    img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2RGB)
    img_load = cv2.resize(img_load, resize)
    cv2.imwrite(resize_img_path, img_load)
    cv2.destroyAllWindows()
    if show_image:
        plt.imshow(cv2.imread(resize_img_path))
        plt.show()
    return img_load


def load_image(image_name, resize_path, origin_path, new_size, show_image=False):
    '''????

    ?????????????????????????????

    Args:
        image_name: ?????
        resize_path: ???????????????
        origin_path: ?????
        new_size: ???????

    Return:
        Array of resized image.
    '''
    resize_image_path = resize_path + '/' + image_name
    origin_image_path = origin_path + '/' + image_name
    if not os.path.exists(resize_image_path):
        resize_image_array = resize_image(resize_path, origin_image_path, new_size, show_image)
    else:
        resize_image_array = image_load(resize_image_path, show_image)
    return resize_image_array

