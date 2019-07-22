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
        mask = cv.resize(mask,(242,152))
        if "gate" in img_path:
            d = np.array(["gate"])
        elif "buoy" in img_path:
            d = np.array(["bouy"])
        else:
            d = np.array(["none"])
        d = np.concatenate((d,mask.ravel()/255.))
        d = ", ".join(map(str, d))
        d += "\n" 
        f.write(d)
    f.close()

def get_dataset(csv_file):
    print("Get Data")

    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []

    f = open(csv_file,"r+")

    for d in f.readlines():
        d = d.split(",")
        x = d[1:]
        y = d[0]
        
        if random.uniform(0, 1) <= .7:
            train_data_x.append(x)
            train_data_y.append(y)
        else:
            test_data_x.append(x)
            test_data_y.append(y)
    print("collected data")
    
    return train_data_x, \
        train_data_y, \
        test_data_x, \
        test_data_y

# if __name__ == "__main__":
#     img2csv("/home/skconan/imgs/labeled")    