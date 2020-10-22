import cv2 as cv
import glob
import numpy as np


def __extract_biggest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Extracts the biggest connected component from a mask (0 and 1's).
    Args:
        img: 2D array of type np.float32 representing the mask

    Returns : 2D array, mask with 1 in the biggest component and 0 outside
    """
    # extract all connected components
    num_labels, labels_im = cv.connectedComponents(mask.astype(np.uint8))

    # we find and return only the biggest one
    max_val, max_idx = 0, -1
    for i in range(1, num_labels):
        area = np.sum(labels_im == i)
        if area > max_val:
            max_val = area
            max_idx = i

    return (labels_im == max_idx).astype(float)


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
    print(count_whites, bright_count, count_all, count_whites / count_all)

    return crop_img, count_whites, count_whites / count_all


def fillied_boxes(im):
    kernel = np.ones((10, 10), np.uint8)
    # im = cv.morphologyEx(im, cv.MORPH_OPEN, kernel)

    h, s, v = cv.split(cv.cvtColor(im, cv.COLOR_BGR2HSV))

    s_in = cv.morphologyEx(s, cv.MORPH_OPEN, kernel)
    _, s_out = cv.threshold(s_in, 30, 250, cv.THRESH_BINARY_INV)
    shape = im.shape
    h = shape[0] // 30 #50
    w = shape[1] // 5
    kernel_eliminate_small = np.ones((h, w), np.uint8)
    s_out = cv.morphologyEx(s_out, cv.MORPH_OPEN, kernel_eliminate_small)
    s_out = cv.morphologyEx(s_out, cv.MORPH_CLOSE, kernel_eliminate_small)

    threshold = 100
    #im = cv.cvtColor(im.astype(np.uint8), cv.COLOR_BGR2RGB)
    kernel_gradient = np.ones((2, 2), np.uint8)
    #gradient = cv.morphologyEx(s_out, cv.MORPH_GRADIENT, kernel_gradient)
    # canny_output = cv.Canny(s_out, threshold, threshold * 2)
    contours, _ = cv.findContours(s_out, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    max_size = 0
    max_fill = 0
    location = [0, 0, 0, 0]

    for i, frame in enumerate(contours):
        a = np.array([tuple(x[0]) for x in frame])
        x, y, w, h = cv.boundingRect(a)
        crop_img, count_whites, fill = check_box_fill(s_out.copy(), x, y, w, h)
        s_out = cv.rectangle(s_out, (x, y), (x + w, y + h), (255, 0, 0), 5)
        im = cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 5)
        print(f'imsize={crop_img.size}, minimal={s_out.size / 70}')

        if crop_img.size > (s_out.size / 70) and h < w:
            if fill > max_fill:
                if (fill == max_fill and crop_img.size > max_size) or max_fill == 0:
                    max_fill = fill
                    location = [x, y, w, h]

    shape = im.shape
    box_img = np.zeros(shape=(shape[0], shape[1]))
    print('location', location)
    box_img[location[1]:location[1] + location[3], location[0]:location[0] + location[2]] = 255
    cv.imwrite('../datasets/masks_extracted/rectangles.png', s_out)

    return box_img, s_out,im


# test()

name = '00002'

im = cv.imread(f'../datasets/qsd1_w2/{name}.jpg')

# output,rect = fillied_boxes(im)
# cv.imwrite(f'../datasets/masks_extracted/{name}_box_mask.png',output)
# cv.imwrite(f'../datasets/masks_extracted/{name}_rect.png', rect)
# quit()
input_files = glob.glob(r'../datasets/qsd1_w2/*.jpg')

for image in input_files:
    im = cv.imread(image)
    name = image[-9:-4]
    print(name)
    output, rect,img_marked = fillied_boxes(im)
    cv.imwrite(f'../datasets/masks_extracted/boxes/{name}_im.png', img_marked)
    cv.imwrite(f'../datasets/masks_extracted/boxes/{name}_box_mask.png', output)
    cv.imwrite(f'../datasets/masks_extracted/boxes/{name}_rect.png', rect)

# cv.imshow('boxes',fillied_boxes(im))
# cv.resizeWindow('boxes', 900,600)
# cv.waitKey()
