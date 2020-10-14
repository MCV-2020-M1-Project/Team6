import argparse, os

import cv2 as cv
import numpy as np
from evaluation import mask_evaluation
import glob
import pickle as pkl
import method_kmeans_colour


def get_measures(name, mask):
    img_annotation = cv.imread(f'../datasets/qsd2_w1/{name}.png', 0)
    p, r, f = mask_evaluation.mask_evaluation(img_annotation, mask)

    measure_dict = {'name': name,
                    'precision': p,
                    'recall': r,
                    'F1_measure': f}

    measure_list = [name, p, r, f]

    return measure_dict


def show_image(im):
    im = cv.imread(f'../datasets/qsd2_w1/{im}.jpg')

    x, y, z = cv.split(im)

    print(type(x))
    cv.imshow('x', x)
    cv.imshow('y', y)
    cv.imshow('z', z)
    print(x)


def method_similar_channels(image, thresh, save, generate_measures):
    # read image into matrix.
    if generate_measures:
        name = image
        img = cv.imread(f'../datasets/qsd2_w1/{name}.jpg').astype(float)  # BGR, float
    else:
        img = image.astype(float)

    # get image properties.
    h, w, bpp = np.shape(img)
    mask_matrix = np.empty(shape=(h, w), dtype='uint8')

    # iterate over the entire image.
    for py in range(0, h):
        for px in range(0, w):
            # print(m[py][px])
            blue = img[py][px][0]
            green = img[py][px][1]
            red = img[py][px][2]
            b_g = blue - green
            b_r = blue - red
            g_r = green - red
            # and bigger than 100 to not be black
            if (-thresh < b_g < thresh) \
                    and (-thresh < b_r < thresh) \
                    and (-thresh < g_r < thresh) \
                    and (blue > 100 and green > 100 and red > 100):
                # print('similar value')
                mask_matrix[py][px] = 0
            else:
                mask_matrix[py][px] = 255
    # cv.imshow('matrix',mask_matrix)
    # cv.waitKey()
    if save:
        if not os.path.exists(f'../datasets/masks_extracted/msc/'):
            os.makedirs(f'../datasets/masks_extracted/msc/')
        cv.imwrite(f'../datasets/masks_extracted/msc/{name}.png', mask_matrix)

    if generate_measures:
        return get_measures(name, mask_matrix)
    else:
        return mask_matrix


def method_colorspace_threshold(image, x_range, y_range, z_range, colorspace, save, generate_measures):
    """
    x = [bottom,top]
    y = [bottom,top]
    z = [bottom,top]

    bottom - top has value from 0-255

    colorspace = 'bgr','rgb','hsv','ycrcb','cielab','
    """
    masks_measures = []
    if generate_measures:
        name = image
        img = cv.imread(f'../datasets/qsd2_w1/{name}.jpg')  # BGR, float
    else:
        img = image

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
    mask_matrix = cv.inRange(img, lower, upper)

    if save:
        if not os.path.exists(f'../datasets/masks_extracted/mst_{colorspace}/'):
            os.makedirs(f'../datasets/masks_extracted/mst_{colorspace}/')
        cv.imwrite(f'../datasets/masks_extracted/mst_{colorspace}/{name}.png', mask_matrix)

    if generate_measures:
        return get_measures(name, mask_matrix)
    else:
        return mask_matrix


def method_mostcommon_color_kmeans(image, k, thresh, colorspace, save, generate_measures):
    """
    methods uses kmeans to find most common colors on the photo, based on this information
    it's filtering that color considering it a background.

    k - provides number of buckets for kmeans algorithm
    thresh - provides number that creates the filter of colors close to the most common one
    colorspcae - allows to choose from different colorspaces bgr to hsv
    save - indicates whether you want to save masks or not

    """


    if generate_measures:
        name = image
        img = cv.imread(f'../datasets/qsd2_w1/{name}.jpg')
    else:
        img = image

    bgr, hsv = method_kmeans_colour.get_most_common_color(img, k)

    if colorspace == 'bgr':
        # mask color
        lower = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
        upper = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
        mask_matrix = cv.inRange(img, lower, upper)
        mask_matrix = cv.bitwise_not(mask_matrix, mask_matrix)

    if colorspace == 'hsv':
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # mask color
        lower = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
        upper = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])
        mask_matrix = cv.inRange(img, lower, upper)
        mask_matrix = cv.bitwise_not(mask_matrix, mask_matrix)

    if save:
        if not os.path.exists(f'../datasets/masks_extracted/mck_{colorspace}/'):
            os.makedirs(f'../datasets/masks_extracted/mck_{colorspace}/')
        cv.imwrite(f'../datasets/masks_extracted/mck_{colorspace}/{name}.png', mask_matrix)

    if generate_measures:
        return get_measures(name, mask_matrix)
    else:
        return mask_matrix


def get_all_methods_per_photo(im, display, save):
    """
    Return a dictionary with all available measures. Keys are:
    * 'msc': method_similar_channels
    * 'mst': method_colorspace_thresholding(image, range x[a,b], range y[c,d], range z[e,f], colorspace)
    """

    measures = {'msc': method_similar_channels(im, 30, save=save, generate_measures=True),
                'mst': method_colorspace_threshold(im, [0, 120], [0, 255], [0, 255], 'bgr', save=save,
                                                   generate_measures=True),
                'msk_bgr': method_mostcommon_color_kmeans(im, 5, 30, colorspace='bgr', save=save,
                                                          generate_measures=True),
                'msk_hsv': method_mostcommon_color_kmeans(im, 5, 10, colorspace='hsv', save=save,
                                                          generate_measures=True)
                }
    # methods returning masks and should return measures
    if display:
        for k in measures.items():
            print(k)

    return measures


def get_all_measures_all_photos(save):
    files_img = glob.glob('../datasets/qsd2_w1/*.png')

    all_measures = []

    for index, image in enumerate(files_img):
        image = image[-9:-4]
        measure = get_all_methods_per_photo(image, display=True, save=save)
        all_measures.append(measure)

    with open('../background_removal_all_methods_all_photos.csv', 'a') as f:
        all_rows = []
        header = 'method' + ',name' + ',precision' + ',recall' + ',F1_measure'
        all_rows.append(header)
        for item in all_measures:
            for k, v in item.items():
                row = "\n" + k
                for i in v.values():
                    row = row + "," + str(i)
                # print(row)
                all_rows.append(row)

        f.writelines(all_rows)

    return all_measures


def main(name, display, save):
    if not os.path.exists('../datasets/masks_extracted/'):
        os.makedirs('../datasets/masks_extracted/')

    if not os.path.exists('pkl_data/masks_extracted/'):
        os.makedirs('pkl_data/masks_extracted/')

    get_all_methods_per_photo(name, display, save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, type=str, help='Name of the imagefile')
    parser.add_argument('-d', '--display', required=False, type=bool, default=True, help='display measures')
    parser.add_argument('-s', '--save', required=False, type=bool, default=False, help='display measures')
    args = parser.parse_args()

    main(args.name, args.display, args.save)

# show_image('00000')
# method_similar_channels('00003')
