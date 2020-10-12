import argparse

import cv2 as cv
import numpy as np
from evaluation import mask_evaluation


def get_measures(image, mask):
    img_annotation = cv.imread(f'../datasets/qsd2_w1/{image}.png',0)
    p, r, f = mask_evaluation.mask_evaluation(img_annotation, mask)

    measure_dict = {'name': image,
                    'precision': p,
                    'recall': r,
                    'F1_measure': f}

    measure_list = [image, p, r, f]

    return measure_dict


def show_image(im):
    im = cv.imread(f'../datasets/qsd2_w1/{im}.jpg')

    x, y, z = cv.split(im)

    print(type(x))
    cv.imshow('x', x)
    cv.imshow('y', y)
    cv.imshow('z', z)
    print(x)


def method_similar_channels(image, thresh):
    # read image into matrix.
    img = cv.imread(f'../datasets/qsd2_w1/{image}.jpg').astype(float)  # BGR, float

    # get image properties.
    h, w, bpp = np.shape(img)
    print(h, w)
    mask_matrix = np.empty(shape=(h, w), dtype='uint8')
    # print('pixel=',mask_matrix[0][0][0])
    # print(mask_matrix)

    thresh = 30

    # iterate over the entire image.
    for py in range(0, h):
        for px in range(0, w):
            # print(m[py][px])
            b_g = img[py][px][0] - img[py][px][1]
            b_r = img[py][px][0] - img[py][px][2]
            g_r = img[py][px][1] - img[py][px][2]
            # and bigger than 100 to not be black
            if (-thresh < b_g < thresh) \
                    and (-thresh < b_r < thresh) \
                    and (-thresh < g_r < thresh):
                # print('similar value')
                mask_matrix[py][px] = 0
            else:
                mask_matrix[py][px] = 255
    # cv.imshow('matrix',mask_matrix)
    # cv.waitKey()

    return get_measures(image,mask_matrix)


def method_colorspace_thresholding(image, x_range, y_range, z_range, colorspace):
    """
    x = [bottom,top]
    y = [bottom,top]
    z = [bottom,top]

    bottom - top has value from 0-255

    colorspace = 'bgr','rgb','hsv','ycrcb','cielab','
    """
    masks_measures = []
    img = cv.imread(f'../datasets/qsd2_w1/{image}.jpg')  # BGR, float

    if colorspace == 'bgr': pass
    if colorspace == 'rgb': img = cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
    if colorspace == 'hsv': img = cv.cvtColor(img, cv.COLOR_BGR2HSV, img)
    if colorspace == 'ycrcb': img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb, img)
    if colorspace == 'cielab': img = cv.cvtColor(img, cv.COLOR_BGR2Lab, img)
    if colorspace == 'xyz': img = cv.cvtColor(img, cv.COLOR_BGR2XYZ, img)
    if colorspace == 'yuv': img = cv.cvtColor(img, cv.COLOR_BGR2YUV, img)

    # mask color
    lower = np.array([x_range[0], y_range[0], z_range[0]])
    upper = np.array([x_range[1], y_range[1], z_range[1]])
    mask0 = cv.inRange(img, lower, upper)

    return get_measures(image,mask0)


def get_all_methods(im, display=False):
    """
    Return a dictionary with all available measures. Keys are:
    * 'msc': method_similar_channels
    * 'mst': method_colorspace_thresholding(image, range x[a,b], range y[c,d], range z[e,f], colorspace)
    """

    measures = {'msc': method_similar_channels(im, 30),
                'mst': method_colorspace_thresholding(im, [0, 120], [0, 255], [0, 255], 'bgr'),
                'other': 'other method',

                }
    #methods returning masks and should return measures
    if display:
        for k in measures.items():
            print(k)

    return measures

def main(image, display):
    get_all_methods(image, display)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, type=str, help='Image')
    parser.add_argument('-d', '--display', required=False, type=bool, default=True, help='display measures')
    args = parser.parse_args()

    main(args.image, args.display)

# show_image('00000')
#method_similar_channels('00003')

