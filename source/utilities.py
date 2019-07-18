#!/usr/bin/env python
"""
    File name: utilities.py
    Author: skconan
    Date created: 2019/04/09
    Python Version: 3.6
"""

import os
import numpy as np
import tensorflow as tf
import colorama
from PIL import Image
from keras.preprocessing import image


colorama.init()
DEBUG = True


def print_debug(*args, **kwargs):
    global DEBUG
    text = ""
    if not "mode" in kwargs:
        mode = "DETAIL"
    else:
        mode = kwargs['mode']
    color_mode = {
        "METHOD": colorama.Fore.BLUE,
        "RETURN": colorama.Fore.GREEN,
        "DETAIL": colorama.Fore.YELLOW,
        "DEBUG": colorama.Fore.RED,
        "END": colorama.Style.RESET_ALL,
    }
    if DEBUG:
        for t in args:
            text += " "+str(t)
        print(color_mode[mode] + text + color_mode["END"])


def get_file_path(dir_name):
    file_list = os.listdir(dir_name)
    files = []
    for f in file_list:
        abs_path = os.path.join(dir_name, f)
        if os.path.isdir(abs_path):
            files = files + get_file_path(abs_path)
        else:
            files.append(abs_path)

    return files


def get_file_name(img_path):
    if "\\" in img_path:
        name = img_path.split('\\')[-1]
    else:
        name = img_path.split('/')[-1]

    name = name.replace('.gif', '')
    name = name.replace('.png', '')
    name = name.replace('.jpg', '')
    return name


def normalize(img):
    result = img / 255.
    return result


def load_image(path, img_rows=256, img_cols=256):
    image_list = np.zeros((len(path),  img_rows, img_cols, 3))
    for i, fig in enumerate(path):
        #try:
        img = image.load_img(fig, target_size=(img_rows, img_cols, 3))
        x = image.img_to_array(img).astype('float32')
        x = normalize(x)
        image_list[i] = x
        #except:
         #   print("error",path[i])
        

    return image_list
