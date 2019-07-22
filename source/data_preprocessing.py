#!/usr/bin/env python
"""
    File name: data_preprocessing.py
    Author: skconan
    Date created: 2019/07/21
    Python Version: 2.7
"""

import pandas as pd
import random
import numpy as np
from utilities import *
import cv2 as cv

def img2csv(dir_path):
    img_path_list = get_file_path(dir_path)
    f = open("./data.csv","w+")
    data = []
    for img_path in img_path_list:
        img = cv.imread(img_path,0)
        _,mask = cv.threshold(
          img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )
        mask = cv.resize(mask,(121,76))
        # cv.imshow("mask",mask)
        # cv.waitKey(10)
        if "gate" in img_path:
            d = np.array([0],np.uint8)
        elif "buoy" in img_path:
            d = np.array([1],np.uint8)
        else:
            d = np.array([2],np.uint8)
        d = np.concatenate((d,img.ravel()/255.))
        d = ", ".join(map(str, d))
        d += "\n" 
        f.write(d)
    # data = np.array(data)
    # np.savetxt("./data.csv", data, delimiter=",")
    f.close()

def get_dataset(csv_file):
    print("Get Data")

    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []

    dataset = pd.read_csv(csv_file, header=None)
    array = np.array(dataset)

    for d in array:
        x = d[1:]
        y = d[0]
   
        if random.uniform(0, 1) <= .7:
            train_data_x.append(x)
            train_data_y.append(y)
        else:
            test_data_x.append(x)
            test_data_y.append(y)

    return train_data_x, \
        train_data_y, \
        test_data_x, \
        test_data_y

if __name__ == "__main__":
    img2csv("/home/skconan/imgs/labeled")    