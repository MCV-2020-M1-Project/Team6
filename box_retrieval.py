import cv2 as cv
import glob
import pickle as pkl
import numpy as np
import os, os.path

def test():

    files_img = glob.glob(f'../datasets/qsd1_w2/*.jpg')

    # im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)

    for i in files_img:
        im = cv.imread(i)
        kernel = np.ones((10, 10), np.uint8)
        # kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(50,1))
        # cv.getStructuringElement(cv.MORPH_RECT,(5,5))
        opening = cv.morphologyEx(im, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel)
        gradient = cv.morphologyEx(im, cv.MORPH_GRADIENT, kernel)

        # opening for dark background white text
        # closing for white background dark text
        # cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        # cv.getStructuringElement(cv.MORPH_RECT,(5,5))

        final = im

        # cv.imshow('opening',opening)
        # cv.waitKey()

        h, s, v = cv.split(cv.cvtColor(im, cv.COLOR_BGR2HSV))
        s_in = s
        h_in = h

        s_in = cv.morphologyEx(s, cv.MORPH_OPEN, kernel)
        h_in = cv.morphologyEx(h, cv.MORPH_OPEN, kernel)

        _, s_out = cv.threshold(s_in, 30, 250, cv.THRESH_BINARY_INV)
        s_out = cv.morphologyEx(s_out, cv.MORPH_OPEN, kernel)
        s_out = cv.morphologyEx(s_out, cv.MORPH_CLOSE, kernel)

        _, h_out = cv.threshold(h_in, 30, 250, cv.THRESH_BINARY_INV)
        h_out = cv.morphologyEx(h_out, cv.MORPH_OPEN, kernel)
        h_out = cv.morphologyEx(h_out, cv.MORPH_CLOSE, kernel)

        # cv.imshow('component',__extract_biggest_connected_component(h_out).astype(np.uint8)*255)
        # cv.waitKey()
        # cv.imshow('connected',components[1].astype(np.uint8))
        # cv.waitKey()

        cv.imwrite(f'../datasets/masks_extracted/morph/{i[-9:-4]}_s_opening_aftersplit.png', s_out)
        cv.imwrite(f'../datasets/masks_extracted/morph/{i[-9:-4]}_h_opening_aftersplit.png', h_out)


def check_box_fill(im, x, y, w, h):
    crop_img = im[y:y + h, x:x + w]
    count_whites = cv.countNonZero(crop_img)
    bright_count = np.sum(np.array(crop_img) >= 1)
    count_all = crop_img.size
    # print(count_whites, bright_count, count_all, count_whites / count_all)

    return crop_img, count_whites, count_whites / count_all


def filled_boxes(im):

    # cv.imshow('im', im)
    # cv.waitKey(0)

    kernel = np.ones((10, 10), np.uint8)
    # im = cv.morphologyEx(im, cv.MORPH_OPEN, kernel)
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # cv.imshow('sat',s)
    # cv.waitKey()

    s_in = cv.morphologyEx(s, cv.MORPH_OPEN, kernel)
    _, s_out = cv.threshold(s_in, 30, 250, cv.THRESH_BINARY_INV)
    # blur = cv.GaussianBlur(s_in, (5, 5), 0)
    #ret3, s_out = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # s_out = cv.adaptiveThreshold(s_in,100,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,5,31)
    shape = im.shape

    h = shape[0] // 50  # 50
    w = shape[1] // 5

    kernel_eliminate_small = np.ones((h, w), np.uint8)
    s_out = cv.morphologyEx(s_out, cv.MORPH_OPEN, kernel_eliminate_small)
    s_out = cv.morphologyEx(s_out, cv.MORPH_CLOSE, kernel_eliminate_small)

    # threshold = 100
    # kernel_gradient = np.ones((2, 2), np.uint8)
    # gradient = cv.morphologyEx(s_out, cv.MORPH_GRADIENT, kernel_gradient)
    # canny_output = cv.Canny(s_out, threshold, threshold * 2)
    contours, _ = cv.findContours(s_out, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    max_fill = 0
    location = [0, 0, 0, 0]

    rect = s_out.copy()

    for i, frame in enumerate(contours):
        a = np.array([tuple(x[0]) for x in frame])
        x, y, w, h = cv.boundingRect(a)
        crop_img, count_whites, fill = check_box_fill(s_out, x, y, w, h)
        rect = cv.rectangle(rect, (x, y), (x + w, y + h), (255, 0, 0), 5)
        im = cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 5)
        # print(f'imsize={crop_img.size}, minimal={s_out.size / 70}')

        if crop_img.size > (s_out.size / 70) and h < w:
            if fill > max_fill:
                max_fill = fill
                location = [x, y, x + w, y + h]

    shape = im.shape
    box_img = np.zeros(shape=(shape[0], shape[1]))
    box_img[location[1]:location[3], location[0]:location[2]] = 1
    cv.imwrite('../datasets/masks_extracted/rectangles.png', rect)

    return s_in, box_img.astype(np.uint8), rect, im, location


def main():
    path = ['..','datasets', 'qsd1_w2', '*.jpg']
    input_files = glob.glob(os.path.join(*path))
    text_boxes_list = create_list_boxes()
    overall_score = []
    score = 0
    all_boxes = []

    for index, image in enumerate(input_files):
        im = cv.imread(image)
        name = image[-9:-4]
        print(name)
        s_in, output, rect, img_marked, location = filled_boxes(im)
        cv.imwrite(f'../datasets/masks_extracted/boxes/{name}_im.png', img_marked)
        cv.imwrite(f'../datasets/masks_extracted/boxes/{name}_box_mask.png', output)
        cv.imwrite(f'../datasets/masks_extracted/boxes/{name}_rect.png', rect)
        cv.imwrite(f'../datasets/masks_extracted/boxes/{name}_sat_morph.png', s_in)

        iou = verify_boxes(location, text_boxes_list[index])
        overall_score.append(iou)
        all_boxes.append(location)

    for i in overall_score:
        score = score + i

    print("overal=", score / len(overall_score))

    path = ['.pkl_data', 'text_boxes.pkl']
    with open(os.path.join(*path), 'wb') as file:
        pkl.dump(obj=all_boxes, file=file)

        # cv.imshow('boxes',fillied_boxes(im))
    # cv.resizeWindow('boxes', 900,600)


def create_list_boxes():
    path = ['..','datasets', 'qsd1_w2', 'text_boxes.pkl']
    with open(os.path.join(*path), 'rb') as file:
        text_boxes_corr = pkl.load(file)

    text_boxes_correspondance = []

    for box in text_boxes_corr:
        x_min, y_min = 10000000000, 10000000000
        x_max, y_max = 0, 0
        for corr in box[0]:
            if corr[0] < x_min: x_min = corr[0]
            if corr[1] < y_min: y_min = corr[1]
            if corr[0] > x_max: x_max = corr[0]
            if corr[1] > y_max: y_max = corr[1]
        # print('min,max=', x_min, y_min, x_max, y_max)
        text_boxes_correspondance.append([x_min, y_min, x_max, y_max])

    # print(text_boxes_correspondance)
    return text_boxes_correspondance


def verify_boxes(location, correspondance):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(location[0], correspondance[0])
    yA = max(location[1], correspondance[1])
    xB = min(location[2], correspondance[2])
    yB = min(location[3], correspondance[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (location[2] - location[0] + 1) * (location[3] - location[1] + 1)
    boxBArea = (correspondance[2] - correspondance[0] + 1) * (correspondance[3] - correspondance[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    print(iou)
    return iou


# main()
