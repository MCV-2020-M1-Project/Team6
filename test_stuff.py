import sys
import pickle as pkl
import cv2
import numpy as np
import descriptor_lib as descs
import distance_metrics_lib as dists
import os, time
import random as rnd
import background_removal as bg


def test_mask_grayish():
    our = bg.method_similar_channels_jc(im2, 30)
    his = bg.method_similar_channels(im2, 30, False)

    cv2.imshow('our',cv2.resize(our*255, (256, 256)))
    cv2.imshow('his',cv2.resize(his, (256, 256)))
    cv2.waitKey(0)


    print(np.sum(abs(our - his)))

def same_hist():
    rnd_img = im1.copy()
    rnd.shuffle(rnd_img)

    a = descs.get_descriptors(im1)
    b = descs.get_descriptors(rnd_img)

    while cv2.waitKey(0) is not ord('q'):
        cv2.imshow('img1',cv2.resize(im1, (256, 256)))
        cv2.imshow('rnd_img',cv2.resize(rnd_img, (256, 256)))
        dists.display_comparison(a['bgr_concat_hist'], b['bgr_concat_hist'])


def morph_threshold_mask(im):
    struct_el = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    im_morph = cv2.morphologyEx(im, cv2.MORPH_CLOSE, struct_el)

    struct_el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    im_morph = cv2.morphologyEx(im_morph, cv2.MORPH_CLOSE, struct_el)

    # struct_el = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # im_morph = cv2.morphologyEx(im_morph, cv2.MORPH_ERODE, struct_el)


    image, contours = cv2.findContours(im_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rect = 0, 0, 0, 0
    max_area = 0
    for cont in image:
        x, y ,w ,h = cv2.boundingRect(cont)
        if w*h > max_area:
            rect = x, y, w, h
            max_area = w*h
    
    mask_im = np.zeros_like(im)
    mask_im[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 1

    return mask_im


def ds_viewer():

    cv2.namedWindow('Before')
    cv2.namedWindow('After')
    # cv2.namedWindow('Dif')
    cv2.namedWindow('Original')

    BD_path = ['..', 'datasets', 'qst2_w1']
    max_num = len([ i for i in os.listdir(os.path.join(*BD_path))  if 'jpg' in i ])
    num = 0
    
    while True:
        img_name = '{:05d}.jpg'.format(int(num))
        im = cv2.imread(os.path.join(*(BD_path + [img_name])), 1)
        print('Img:', img_name)
        # mask_matrix = bg.method_colorspace_threshold(im.copy(), [0, 255], [100, 255], [0, 200], 'hsv') # kinda working
        # mask_im = morph_threshold_mask(mask_matrix)

        # mask_matrix = bg.method_colorspace_threshold(im.copy(),  [4, 24], [163, 203],[163, 223], 'bgr')
        # mask_im = morph_threshold_mask(mask_matrix)

        s = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1]
        mask_matrix = cv2.morphologyEx(s, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
        mask_im = cv2.threshold(mask_matrix, 30, 255, cv2.THRESH_BINARY)
        
        cv2.imshow('Before', cv2.resize(255*mask_matrix, (512, int(512*mask_matrix.shape[0]/mask_matrix.shape[1]))))

        cv2.imshow('After', cv2.resize(255*mask_im, (512, int(512*mask_im.shape[0]/mask_im.shape[1]))))
        cv2.imshow('Original', cv2.resize(im, (512, int(512*im.shape[0]/im.shape[1]))))

        cv2.moveWindow('Original', 0, 0)
        cv2.moveWindow('Before', 600, 0)
        cv2.moveWindow('After', 1150, 0)

        char = cv2.waitKey(0)

        if char is ord('p') or char is 81 and num > 0:
            num -= 1
        elif char is ord('n') or char is 83 and num < max_num - 1:
            num += 1
        elif char is ord('q'):
            break
        else:
            # print(char)
            continue


        
        





    # mask_matrix2, contours, hierarchy = cv2.findContours(mask_matrix, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


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

    # BD_path = ['..', 'datasets']

    # BD_img = 'bbdd_00009.jpg'
    # query_img = '{:05d}.jpg'.format(int(sys.argv[1])) # '00004.jpg'

    # im1 = cv2.imread(os.path.join(*BD_path, 'BBDD', BD_img), 1)
    # im2 = cv2.imread(os.path.join(*BD_path, 'qsd2_w1', query_img), 1)

    # distances()
    # visualize_hist()
    # same_hist()
    # test_mask_grayish()
    ds_viewer()
