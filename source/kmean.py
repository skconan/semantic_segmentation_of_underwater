import cv2 as cv
from lib import *
import numpy as np
from bg_subtraction import bg_subtraction, bg_subtraction_gray


def kmean(img, k=8):
    z = img.reshape((-1, 3))

    z = np.float32(z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(
        z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))
    return res

def find_color(img):
    img_flat = img.copy().ravel()
    color = []
    color_list = []
    for i in img_flat:
        color_list.append(i)
        if len(color_list) == 3:
            color.append(color_list)
            color_list = []
    color = set(tuple(i) for i in color)
    color = list(color)

    return color

def main():
    path = r"C:\Users\skconan\Desktop\cycleGANs\dataset_new\gt_kmean_test - Copy"
    path_gate_00 = r"C:\Users\skconan\Desktop\cycleGANs\dataset_new\groundTruth_kmean_test"
    out_path = r"C:\Users\skconan\Desktop\cycleGANs\dataset_new\groundTruth_kmean_mask"
    # path_gate_01 = r"C:\Users\skconan\Desktop\underwater_object_detection\dataset\gate\01"
    img_list = get_filename(path_gate_00)
    for img_path in img_list:
        img = cv.imread(img_path)
        name = get_img_name(img_path)
        if not os.path.exists(path +"\\" + name + ".jpg"):
            continue
        # img = cv.resize(img,None,fx=0.3,fy=0.3)
        # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # h,s,v = cv.split(hsv)
        # result = kmean(img)
        # neg, pos = bg_subtraction_gray(h,61,1)
        # neg, pos = bg_subtraction(h,21,1)
        # imshow("image",img)
        # imshow("hue",h)
        # imshow("sub-neg",neg)
        # imshow("sub-pos",pos)
        
        # img = cv.bitwise_and(img,img,mask=255-neg)
        
        # res_k_mean = kmean(img)
        # hsv_k_mean = cv.cvtColor(res_k_mean,cv.COLOR_BGR2HSV)

        # color = find_color(hsv_k_mean)
        # color.remove((0,0,0))

        # result_rm_k_mean = np.zeros(h.shape,np.uint8)
        # for c in color:
        #     # lower = [c[0],c[1],c[2]]
        #     lower = [c[0],c[1]-10,c[2]-10]
        #     upper = [c[0],c[1]+10,c[2]+10]
    
        #     lower = np.array(lower, np.uint8)
        #     upper = np.array(upper, np.uint8)
    
        #     in_range = cv.inRange(hsv,lower,upper)
        #     result_rm_k_mean = cv.bitwise_or(result_rm_k_m
        # lower = np.array([0,230,230],np.uint8)
        # upper = np.array([40,255,255],np.uint8)
        # res = cv.inRange(img,lower,upper)
        # cv.imwrite(out_path + "\\" + name + ".jpg",img)
        try:
            os.remove(r"C:\Users\skconan\Desktop\cycleGANs\dataset_new\groundTruth_kmean_train"+"\\" + name + ".jpg")
        except:
            pass
        # os.remove(r"C:\Users\skconan\Desktop\cycleGANs\dataset_new\images_train"+"\\" + name + ".jpg")
        # k = cv.waitKey(10) & 0xff
        # if k == ord('q'):
        #     cv.destroyAllWindows()
        #     break  
if __name__ == "__main__":
    main()