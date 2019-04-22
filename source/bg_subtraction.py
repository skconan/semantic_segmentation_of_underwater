#!/usr/bin/env python
"""
    File name: bg_subtraction.py
    Author: skconan
    Date created: 2010/01/10
    Python Version: 3.6
"""
import constans as CONST
import numpy as np
import cv2 as cv
from utilities import *
import time


def bg_subtraction(bgr, k_size_bg=61, k_size_fg=5):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    bg = cv.medianBlur(gray, 61)
    fg = cv.medianBlur(gray, 5)

    sub_sign = np.int16(fg) - np.int16(bg)

    sub_pos = np.clip(sub_sign.copy(), 0, sub_sign.copy().max())
    sub_pos = normalize(sub_pos)

    sub_neg = np.clip(sub_sign.copy(), sub_sign.copy().min(), 0)
    sub_neg = normalize(sub_neg)

    _, obj_neg = cv.threshold(
        sub_neg, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )

    _, obj_pos = cv.threshold(
        sub_pos, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    cv.imshow("bg", bg)
    cv.imshow("fg", fg)
    return obj_neg, obj_pos


def bg_subtraction_gray(gray, k_size_bg=61, k_size_fg=5):
    bg = cv.medianBlur(gray, k_size_bg)
    fg = cv.medianBlur(gray, k_size_fg)

    sub_sign = np.int16(fg) - np.int16(bg)

    sub_pos = np.clip(sub_sign.copy(), 0, sub_sign.copy().max())
    sub_pos = normalize(sub_pos)

    sub_neg = np.clip(sub_sign.copy(), sub_sign.copy().min(), 0)
    sub_neg = normalize(sub_neg)

    _, obj_neg = cv.threshold(
        sub_neg, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )

    _, obj_pos = cv.threshold(
        sub_pos, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    cv.imshow("bg", bg)
    cv.imshow("fg", fg)
    return obj_neg, obj_pos


def main():
    cap = cv.VideoCapture(CONST.PATH_VDO + r'\robosub_00.mp4')
    print_debug(cap.isOpened())
    surf = cv.cv2.xfeatures2d.SURF_create(500)
    # while cap.isOpened():
    file_path_list = get_filename(
        r"C:\Users\skconan\Desktop\underwater_object_detection\dataset\robosub")
    for img_path in file_path_list:
        frame = cv.imread(img_path, 1)
        # ret, frame = cap.read()

        # if not ret:
        #     print_debug("Cannot read frame")
        #     break
        if frame is None:
            print_debug("Cannot read frame")
            break
        frame = cv.resize(frame.copy(), None, fx=0.2, fy=0.2)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # for i in range(31,32,20):
        #     k_size_bg = i
        #     k_size_fg = 3
        #     print_debug("k size bg:",k_size_bg)

        #     sub_neg, sub_pos = bg_subtraction(frame,k_size_bg, k_size_fg)
        #     imshow("sub_neg_"+str(k_size_bg), sub_neg.copy())
        #     imshow("sub_pos_"+str(k_size_bg), sub_pos.copy())
        #     k = cv.waitKey(1) & 0xff
        k_size_bg = 61
        k_size_fg = 3
        # print_debug("k size bg:",k_size_bg)
        # sub_neg, sub_pos = bg_subtraction(frame,k_size_bg, k_size_fg)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        sub_neg, sub_pos = bg_subtraction_gray(v, k_size_bg, k_size_fg)
        imshow("sub_neg_"+str(k_size_bg), sub_neg.copy())
        imshow("sub_pos_"+str(k_size_bg), sub_pos.copy())
        sub_pos = 255 - sub_pos
        sub = cv.bitwise_or(sub_pos, sub_neg)
        rect = get_kernel(ksize=(5, 5))
        sub = cv.dilate(sub, rect)
        img = frame.copy()
        img[sub == 255] = 0
        bg = cv.inpaint(img, sub, 11, cv.INPAINT_TELEA)
        imshow("sub", sub)
        imshow("sub_pos_inv", sub_pos)
        imshow("img", img)
        imshow("bg_res", bg)
        gray_bg = cv.cvtColor(bg, cv.COLOR_BGR2GRAY)
        # sub_neg_res, sub_pos_res = bg_subtraction_gray(gray,k_size_bg, k_size_fg)
        sub_res = np.int16(gray) - np.int16(gray_bg)
        sub_res = np.abs(sub_res)
        sub_res[sub_res <= 20] = 0
        # _,th_res = cv.thres
        sub_res = sub_res*255/sub_res.max()
        sub_res[sub_res > 20] = 255
        sub_res = np.uint8(sub_res)
        imshow("sub_res", np.uint8(sub_res))
        # imshow("sub_pos_res", sub_pos_res)
        # orb = cv.ORB_create(nfeatures=1500)
        keypoints, descriptors = surf.detectAndCompute(gray, None)
        print(descriptors)
        frame = cv.drawKeypoints(frame, keypoints, None, (255, 0, 255), 4)
        keypoints, descriptors = surf.detectAndCompute(bg, None)
        frame = cv.drawKeypoints(frame, keypoints, None, (0, 255, 255), 4)
        imshow("original", frame)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
