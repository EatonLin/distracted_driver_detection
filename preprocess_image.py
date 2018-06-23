import shutil
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelBinarizer
import cv2
import matplotlib.pyplot as plt
import os
import re
import random

outliers_dict = {}


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


def get_image_number(image_name):
    '''?????????ID

    Args:
        image_name: ?????

    Returns:
        ??ID
    '''
    return re.split(r"[_.]", image_name)


def is_outlier(outliers_dict, image_name, class_name):
    '''???????????????????

    Args:
        outliers_dict: ?????
        image_name: ?????
        class_name: ???????

    Returns:
        true: ???????, ????????
    '''
    image_number = get_image_number(image_name)
    image_array = outliers_dict[class_name]
    return image_number in image_array


def load_driver_feature(image_names, class_names, resize_path, origin_path, resize, show_image=False):
    '''

    Args:
        image_names:
        class_names:
        resize_path:
        origin_path:
        resize:
        show_image:

    Returns:

    '''
    image_array = []
    for image_name, class_name in zip(image_names, class_names):
        if is_outlier(image_name, class_name):
            continue
        resize_class_path = resize_path + '/' + class_name
        origin_class_path = origin_path + '/' + class_name
        image_array.append(load_image(image_name, resize_class_path, origin_class_path, resize, show_image))
    return image_array


def append_outliers_to_dict(classname, outliers):
    '''?????????????

    Args:
        classname: ?????
        outliers: ???

    Returns:
        None
    '''
    outliers_dict[classname] = outliers


def get_outliers_dict():
    return outliers_dict


def is_outlier(image_name, class_name):
    '''???????????????????

    Args:
        image_name: ?????
        class_name: ???????

    Returns:
        true: ???????, ????????
    '''
    image_number = get_image_number(image_name)
    image_array = outliers_dict[class_name]
    return image_number in image_array


def load_driver_feature(image_names, class_names, resize_path, origin_path, resize):
    '''????????????????????????

    Args:
        image_names: ????
        class_names: ????
        resize_path: ??????????
        origin_path: ????????
        resize: ????????

    Returns:
        ??????????????
    '''
    image_array = []
    for image_name, class_name in zip(image_names, class_names):
        if is_outlier(image_name, class_name):
            continue
        resize_class_path = resize_path + '/' + class_name
        origin_class_path = origin_path + '/' + class_name
        image_array.append(load_image(image_name, resize_class_path, origin_class_path, resize))
    return image_array


def load_feature_label(origin_df, driver, resize_path, origin_path, resize):
    '''???driver id??????????????????????????????

    Args:
        origin_df: ?????????????
        driver: ????
        resize_path: ????????????
        origin_path: ?????
        resize: ?????????

    Returns:
        ????driver????????????????
    '''
    features = []
    labels = []
    classes = origin_df[origin_df['subject'] == driver].classname.values
    images = origin_df[origin_df['subject'] == driver].img.values
    image_array = load_driver_feature(
        images,
        classes,
        resize_path,
        origin_path,
        resize)
    features.extend(image_array)
    labels.extend(one_hot_encode(classes))
    return features, labels


def train_valid_split(origin_df, valid_size, origin_path, resize_train_path, resize_valid_path, resize):
    '''你大爷

    Args:
        origin_df: ?????????????
        valid_size: ???:??????
        resize_path: ?????????????
        origin_path: ?????
        resize: ????????

    Returns:
        ???????????????
    '''
    feature_train = []
    label_train = []
    feature_valid = []
    label_valid = []
    drivers = list(set(origin_df['subject']))
    random.shuffle(drivers)
    total = len(drivers)
    valid_total = int(valid_size * total)
    train_total = total - valid_total
    print("Drivers size:%d valid_total:%d" % (len(drivers), valid_total))

    for driver in drivers[:train_total]:
        # print("Loading driver id:%s as train data" % driver)
        features, labels = load_feature_label(origin_df, driver, resize_train_path, origin_path, resize)
        feature_train.extend(features)
        label_train.extend(labels)
        # print("Loading finish. feature_train size:%d, label_train size:%d" % (len(feature_train), len(label_train)))

    for driver in drivers[train_total:]:
        # print("Loading driver id:%s as valid data" % driver)
        features, labels = load_feature_label(origin_df, driver, resize_valid_path, origin_path, resize)
        feature_valid.extend(features)
        label_valid.extend(labels)
        # print("Loading finish. feature_valid size:%d, label_valid size:%d" % (len(feature_valid), len(label_valid)))

    print("Loading completed")
    return feature_train, label_train, feature_valid, label_valid


def copy_image(image_name, class_name, origin_path, target_path):
    new_path = target_path + '/' + class_name
    new_image_path = new_path + '/' + image_name
    src_path = origin_path + '/' + class_name
    src_image_path = src_path + '/' + image_name

    # print(image_name, class_name, origin_path, target_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    shutil.copyfile(src_image_path, new_image_path)



def splite_train_valid(train_df, valid_size, origin_path, resize_train_path, resize_valid_path):
    drivers = list(set(train_df['subject']))
    random.shuffle(drivers)
    total = len(drivers)
    train_total = total - int(valid_size * total)

    print("--------Start spliting-----------")

    for driver in drivers[:train_total]:
        classes = train_df[train_df['subject'] == driver].classname.values
        images = train_df[train_df['subject'] == driver].img.values
        for class_name, image_name in zip(classes, images):
            copy_image(image_name, class_name, origin_path, resize_train_path)

    for driver in drivers[train_total:]:
        classes = train_df[train_df['subject'] == driver].classname.values
        images = train_df[train_df['subject'] == driver].img.values
        for class_name, image_name in zip(classes, images):
            copy_image(image_name, class_name, origin_path, resize_train_path)

    print("--------Spliting completed-----------")
