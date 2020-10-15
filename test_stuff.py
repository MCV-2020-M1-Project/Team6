import pickle as pkl
import cv2
import numpy as np
import descriptor_lib as descs
import distance_metrics_lib as dists
import os, time
import random as rnd


def same_hist():
    rnd_img = im1.copy()
    rnd.shuffle(rnd_img)

    a = descs.get_descriptors(im1)
    b = descs.get_descriptors(rnd_img)

    while cv2.waitKey(0) is not ord('c'):
        cv2.imshow('img1',cv2.resize(im1, (256, 256)))
        cv2.imshow('rnd_img',cv2.resize(rnd_img, (256, 256)))
        dists.display_comparison(a['bgr_concat_hist'], b['bgr_concat_hist'])



def distances():

    cv2.imshow('img1',cv2.resize(im1, (256, 256)))
    cv2.imshow('img2',cv2.resize(im2, (256, 256)))


    a = descs.get_descriptors(im1)
    b = descs.get_descriptors(im2)
    
    DESCR = 'hsv_concat_hist'

    while cv2.waitKey(0) is not ord('c'):
        dists.display_comparison(a[DESCR], b[DESCR])


def visualize_hist():
    new_img = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

    for c, n in enumerate(['H', 'S', 'V']):
        cv2.imshow(n, new_img[:, :, c])

    cv2.waitKey(0)

if __name__ == '__main__':

    BD_path = ['..', 'datasets']

    BD_img = 'bbdd_00009.jpg'
    query_img = '00004.jpg'

    im1 = cv2.imread(os.path.join(*BD_path, 'BBDD', BD_img), 1)
    im2 = cv2.imread(os.path.join(*BD_path, 'qsd1_w1', query_img), 1)

    # distances()
    # visualize_hist()
    same_hist()