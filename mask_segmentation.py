import cv2 as cv
import numpy as np
from evaluation import mask_evaluation
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def show_xyz(im, colorspace):
    """
    im - number from the name of the file
    colorspace
    """

    im = cv.imread(f'../datasets/qsd2_w1/{im}.jpg')

    if colorspace == 'bgr': pass
    if colorspace == "gray": im = cv.cvtColor(im, cv.COLOR_BGR2GRAY, im)
    if colorspace == 'rgb': im = cv.cvtColor(im, cv.COLOR_BGR2RGB, im)
    if colorspace == 'hsv': im = cv.cvtColor(im, cv.COLOR_BGR2HSV, im)
    if colorspace == 'ycrcb': im = cv.cvtColor(im, cv.COLOR_BGR2YCrCb, im)
    if colorspace == 'cielab': im = cv.cvtColor(im, cv.COLOR_BGR2Lab, im)
    if colorspace == 'xyz': im = cv.cvtColor(im, cv.COLOR_BGR2XYZ, im)
    if colorspace == 'yuv': im = cv.cvtColor(im, cv.COLOR_BGR2YUV, im)

    if colorspace != "gray":
        x, y, z = cv.split(im)
        cv.imshow('x', x)
        cv.imshow('y', y)
        cv.imshow('z', z)
    else:
        cv.imshow('gray', im)

    cv.waitKey(0)
    cv.destroyAllWindows()


def generate_masks(x_range, y_range, z_range, colorspace):
    """
    x = [bottom,top]
    y = [bottom,top]
    z = [bottom,top]

    bottom - top has value from 0-255

    colorspace = 'bgr','rgb','hsv','ycrcb','cielab','
    """
    masks_measures = []
    files_img = glob.glob('../datasets/qsd2_w1/*.png')

    for index, obraz in enumerate(files_img):
        im = cv.imread(obraz[:-3] + "jpg")

        if colorspace == 'bgr': pass
        if colorspace == 'rgb': im = cv.cvtColor(im, cv.COLOR_BGR2RGB, im)
        if colorspace == 'hsv': im = cv.cvtColor(im, cv.COLOR_BGR2HSV, im)
        if colorspace == 'ycrcb': im = cv.cvtColor(im, cv.COLOR_BGR2YCrCb, im)
        if colorspace == 'cielab': im = cv.cvtColor(im, cv.COLOR_BGR2Lab, im)
        if colorspace == 'xyz': im = cv.cvtColor(im, cv.COLOR_BGR2XYZ, im)
        if colorspace == 'yuv': im = cv.cvtColor(im, cv.COLOR_BGR2YUV, im)

        im_annotation = cv.imread(obraz, 0)
        # mask color
        lower = np.array([x_range[0], y_range[0], z_range[0]])
        upper = np.array([x_range[1], y_range[1], z_range[1]])
        mask0 = cv.inRange(im, lower, upper)

        mask0 = cv.bitwise_not(mask0)
        # cv.imshow('mask',mask0)

        if not os.path.exists('../datasets/masks_extracted/'):
            os.makedirs('../datasets/masks_extracted/')

        #cv.imwrite('../datasets/masks_extracted/' + obraz[-9:], mask0)

        p, r, f = mask_evaluation.mask_evaluation(im_annotation, mask0)

        measure_dict = {'name': index,
                        'precision': p,
                        'recall': r,
                        'F1_measure': f}

        measure_list = [index, p, r, f]
        masks_measures.append(measure_list)

    return masks_measures


def generate_measures_output(input_measures_data, show_graph=False):
    """
    show_graph True if you want to see scatterplot of measures
    measure data as list of lists

    """
    df = pd.DataFrame(np.array(input_measures_data), columns=['image', 'precision', 'recall', 'f1'])
    df.head()
    df.to_excel("../datasets/masks_extracted/output_measures.xlsx")

    sns.relplot(data=df,
                y='image',
                x='precision',
                aspect=2.5).set_titles('precision')

    sns.relplot(data=df,
                y='image',
                x='recall',
                aspect=2.5).set_titles('recall')

    sns.relplot(data=df,
                y='image',
                x='f1',
                aspect=2.5).set_titles('f1')

    thresh = 0.8

    print("Treshold = ", thresh)
    print("----------------------ABOVE TRESHOLD--------------------")
    print(df.f1[df['f1'] > thresh].count())
    print("-----------------------BELOW TRESHOLD--------------------")
    print(df.f1[df['f1'] < thresh].count())
    print("----------------------LIST ABOVE TRESHOLD--------------------")
    print(df[df['f1'] > thresh])
    print("-------------AVERAGE PRECISION------------------------")
    print(df['precision'].mean())
    print("-------------AVERAGE RECALL------------------------")
    print(df['recall'].mean())
    print("-------------AVERAGE F1------------------------")
    print(df['f1'].mean())

    if show_graph: plt.show()


def balance_white_image(im):
    image = cv.imread(f'../datasets/qsd2_w1/{im}.jpg')

    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    image = wb.balanceWhite(image)

    return image


def equalize_histogram(im):
    im = cv.imread(f'../datasets/qsd2_w1/{im}.jpg')
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(im)

    return dst


# img = cv.imread('../datasets/qsd2_w1/00004.jpg')
# img_annotation = cv.imread('../datasets/qsd2_w1/00004.png',0)

# show_xyz('00004','yuv')

# measures_data = generate_masks([0,20],[150,255],[0,255], 'hsv')
#measures_data = generate_masks([50,255],[120,255],[50,255], 'bgr')
# measures_data = generate_masks([100,255],[0,255],[0,255], 'cielab')
#measures_data = generate_masks([50, 255], [0, 255], [100, 255], 'ycrcb')

#generate_measures_output(measures_data, False)

# cv.imshow('eqlz',equalize_histogram('00024'))
# cv.waitKey()
# show_xyz('00017','ycrcb')
# cv.waitKey()

"""
## what if we just take histogram of center of the photo where for sure is
## and compare against other centers

#ycrb works great for picture 24 and 10 - why ?
#blue cut out at 108-134 is best with red and green equal to 0 from all channels

how well it will find cut out photos with 0,79 F1 ?
what if we take hue channel 0-20 where precision in 86% and low recall and connect with blue channel over 150
which has high recall over 90%

"""

def generate_report():

    all_mean_measures=[]

    for i in range(0,255):
        input_measures_data = generate_masks([0,255],[i,255],[0,255],'bgr')
        df = pd.DataFrame(np.array(input_measures_data), columns=['image', 'precision', 'recall', 'f1'])

        mean_measures = [i, df['precision'].mean(),df['recall'].mean(), df['f1'].mean()]
        all_mean_measures.append((mean_measures))
        print(i)

    df_all = pd.DataFrame(np.array(all_mean_measures), columns=['green value', 'precision', 'recall', 'f1'])
    df_all.to_excel("../output_measures_bgr_g_notbitwisenot.xlsx")

generate_report()