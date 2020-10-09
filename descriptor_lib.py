import cv2
import numpy as np

def get_gray_hist(img):
    '''Paramenters: img (color image)
        Returns: hist (grayscale histogram) '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    return hist

def get_bgr_concat_hist(img):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 bgr histograms concatenated '''

    hist_b = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_r = cv2.calcHist([img],[2],None,[256],[0,256])
    return np.concatenate((hist_b, hist_g, hist_r))

def get_cielab_concat_hist(img):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 lab histograms concatenated '''

    cielab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    hist_l = cv2.calcHist([cielab],[0],None,[256],[0,256])
    hist_a = cv2.calcHist([cielab],[1],None,[256],[0,256])
    hist_b = cv2.calcHist([cielab],[2],None,[256],[0,256])
    return np.concatenate((hist_l, hist_a, hist_b))

def get_ycrcb_concat_hist(img):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 YCrCb histograms concatenated '''

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hist_y = cv2.calcHist([ycrcb],[0],None,[256],[0,256])
    hist_cr = cv2.calcHist([ycrcb],[1],None,[256],[0,256])
    hist_cb = cv2.calcHist([ycrcb],[2],None,[256],[0,256])
    return np.concatenate((hist_y, hist_cr, hist_cb))

def get_hsv_concat_hist(img):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 HSV histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv],[0],None,[256],[0,256])
    hist_s = cv2.calcHist([hsv],[1],None,[256],[0,256])
    hist_v = cv2.calcHist([hsv],[2],None,[256],[0,256])
    return np.concatenate((hist_h, hist_s, hist_v))

def get_descriptors(img):
    ''' Paramenters: img (color image)
        Returns: descript_dic (dictionary with descriptors names as keys) '''

    descript_dic = {}
    descript_dic['gray_hist'] = get_gray_hist(img)
    descript_dic['bgr_concat_hist'] = get_bgr_concat_hist(img)
    descript_dic['cielab_concat_hist'] = get_bgr_concat_hist(img)
    descript_dic['ycrcb_concat_hist'] = get_ycrcb_concat_hist(img)
    descript_dic['hsv_concat_hist'] = get_hsv_concat_hist(img)
    return descript_dic