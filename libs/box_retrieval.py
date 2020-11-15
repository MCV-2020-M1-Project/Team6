import os
import sys
sys.path.append(os.path.split(__file__)[0])

import cv2 as cv
import glob
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

import descriptors

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

def get_boxes(im):

    original_size = im.shape[:2]
    original_image = im.copy()

    im = cv.resize(im, (500, 500*im.shape[0]//im.shape[1]))
    him, wim = im.shape[:2]

    # cv.imshow('painting:',im)
    # cv.waitKey(0)
    
    im = linear_stretch( im, descriptors.get_bgr_concat_hist(im))

    #gradients
    kernel = np.ones((7,7),np.float32)/49
    blur = cv.filter2D(im,-1,kernel)

    laplacian = cv.Laplacian(blur,cv.CV_64F)
    extrem = (abs(np.min(laplacian))+abs(np.max(laplacian)))//2
    tol = extrem*0.35

    lap = 255 * np.uint8(abs(laplacian[:, :, 0])>tol) # * np.uint8(abs(laplacian[:, :, 1])>tol) * np.uint8(abs(laplacian[:, :,2])>tol)
    
    mask = np.zeros_like(lap)    
    mask[int(him*0.05):-int(him*0.05),int(wim*0.2):-int(wim*0.2)] = lap[int(him*0.05):-int(him*0.05),int(wim*0.2):-int(wim*0.2)]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (wim//5,4))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,6))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (wim//10,1))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    draw = mask.copy()
    rectangles=[]
    for blob in contours:
        x, y, w, h = cv.boundingRect(blob)
        fill = check_box_fill(mask[:,:,0], x, y, w, h)[2]
        if len(contours) == 1:
            rectangles.append((x,y,w,h))
            cv.rectangle(draw, (x,y), (x+w,y+h), (0,0,255))
        elif h>5 and w*h<wim*him*0.35 and h/w<0.9 and fill > 0.35:
            cx = x+w//2
            cy = y+h//2
            if wim*0.2>abs(wim//2-cx) and him*0.12<abs(him//2-cy):
                rectangles.append((x,y,w,h))
                cv.rectangle(draw, (x,y), (x+w,y+h), (0,0,255))
    # cv.imshow('rectangles hold filters:',draw)
    # cv.waitKey(0)

    # cv.imshow('rects',draw)
    # cv.waitKey(0)

    rectangles = sorted(rectangles, key=lambda x: x[1])
    aux = []
    if len(rectangles) == 1:
        aux = rectangles
    else:
        i = 0
        while i < len(rectangles)-1:
            if abs(rectangles[i][1]+rectangles[i][3]-rectangles[i+1][1])<15:
                y = rectangles[i][1] 
                x = rectangles[i][0] if rectangles[i][0] < rectangles[i+1][0] else rectangles[i+1][0]
                h = abs(y - rectangles[i+1][1]-rectangles[i+1][3])
                w = rectangles[i][2] if rectangles[i][2] > rectangles[i+1][2] else rectangles[i+1][2]
                aux.append((x,y,w,h))
                i+=1
            else:
                aux.append(rectangles[i])
            if i+1 == len(rectangles)-1:
                aux.append(rectangles[i+1])
            i+=1

    for rect in aux:
        cv.rectangle(draw, (rect[0],rect[1]), (rect[0]+rect[2],rect[3]+rect[1]), (0,255,0))
    # cv.imshow('post join:',draw)
    # cv.waitKey(0)

    if len(aux) < 1:
        return None

    the_rect = aux[0] 
    if len(aux)>1:
        the_rect = choose_best_rectangle(aux, laplacian)
    
    #expand_rect(aux[0])

    box_mask = np.zeros_like(mask[:,:,0])
    box_mask[the_rect[1]:the_rect[1]+the_rect[3], the_rect[0]:the_rect[0]+the_rect[2]] = np.ones((the_rect[3],the_rect[2]), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (wim//25,10))
    box_mask = cv.morphologyEx(box_mask, cv.MORPH_DILATE, kernel, iterations=2)


    box_mask = cv.resize(box_mask, original_size[::-1])
    
    # a = np.where(box_mask > 0)
    # pts = [(i, j) for i,j in zip(*a)]
    # if len(pts) > 0:
    #     box_mask = fit_box(pts, original_image)
  

    return box_mask

def fit_box(rect, im):

    def get_xy(event, x, y, *etc):
        '''
        Used for getting pixel coordinates in imshow, just for debugging
        '''
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
    
    def is_black_over_white(im):
        '''
        Checks whether the input (binarised) images is black over white or the other way around
        '''
        im_area = im.shape[0]*im.shape[1]
        num_white = np.sum(im/255)

        # print(100*num_white/im_area, '% white')
        return  100*num_white/im_area > 50

    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY) if len(im.shape) == 3 else im
    im = cv.GaussianBlur(im,(3,3),0)
    box = im[rect[0][0]:rect[-1][0], rect[0][1]:rect[-1][1]]
    # cv.imshow('before', box)

    # print(im.shape)
    # print(rect[0][0], rect[-1][0], rect[0][1], rect[-1][1])
    # print(box.shape)
    gray = cv.threshold(box, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    if not is_black_over_white(gray):
        gray = cv.bitwise_not(gray)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)))
    gray_sob = cv.Sobel(gray.copy(), cv.CV_64F, 1, 0)
    
    vector = gray[gray.shape[0]//2, :].copy()
    # cv.normalize(vector,vector,1,norm_type=cv.NORM_L2)

    gmax =vector.max()
    
    for i in range(len(vector)):
        if abs(vector[i]) < gmax*0.5:
            vector[i] = 0
    vector = abs(vector)

    grad_vector = np.float64(vector.copy())
    grad_vector[:-1] = abs(grad_vector[1:] - grad_vector[:-1])
    grad_vector[-1] = 1 # dirty trick
    widths = []
    d = 0
    for i in range(len(grad_vector)):
        if grad_vector[i] > 0:
            widths.append(i - d)
            d = i
    med = np.median(widths)

    # print(med)
    # print(widths)
    # win_size = 5
    # pos = 0
    # for i in range(len(widths)-win_size):
    #     pos += widths[i]
    #     if np.sum(widths[i:i+win_size] < med - 4) > 2:
    #         print('match at', pos)

    # plt.plot(vector)
    # plt.plot(grad_vector)
    gray = abs(gray)
    gray = gray.astype(np.uint8)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.line(gray, (0,gray.shape[0]//2),  (gray.shape[1],gray.shape[0]//2), (255,0,0))
    # cv.namedWindow('gray')
    # cv.setMouseCallback('gray', get_xy)
    # cv.imshow('gray', gray)
    beg = widths[0] - 5 if widths[0] > 5 else widths[0]
    end = -widths[-1] + 10 if widths[-1] > 10 else -widths[-1]
    # plt.show()
    # cv.waitKey(0)
    the_rect = rect[0][0], rect[-1][0], rect[0][1]+beg, rect[-1][1]+end
    ret_mask = np.zeros_like(im)
    ret_mask[the_rect[0]:the_rect[1], the_rect[2]:the_rect[3]] = np.ones((the_rect[1] - the_rect[0],the_rect[3]-the_rect[2]), np.uint8)
    return ret_mask

def choose_best_rectangle(rects, laplacian):
    the_rect = None
    min_sumita = np.inf
    for rect in rects:
        again = laplacian[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        vector = again[again.shape[0]//2, :, :].copy()
        cv.normalize(vector,vector,1,norm_type=cv.NORM_L2)

        vector[:,0] = [vector[i,0]/abs(vector[:,0].min()) if vector[i,0]<0 else vector[i,0]/vector[:,0].max() for i in range(len(vector[:,0]))]
        vector[:,1] = [vector[i,1]/abs(vector[:,0].min()) if vector[i,1]<0 else vector[i,1]/vector[:,0].max() for i in range(len(vector[:,1]))]
        vector[:,2] = [vector[i,2]/abs(vector[:,0].min()) if vector[i,2]<0 else vector[i,2]/vector[:,0].max() for i in range(len(vector[:,2]))]

        sumita = np.sum(abs(vector[:,0]-vector[:,1]) + abs(vector[:,0]-vector[:,2]) + abs(vector[:,1]-vector[:,2]))/len(vector)
        if sumita < min_sumita:
            the_rect = rect
            min_sumita = sumita
        # print(sumita)
        # print(vector.shape)

        # cv.imshow('a ver:', again*255)
        # plt.plot(vector)
        # plt.show()
        # plt.plot(again[again.shape[0]//2, :])
        # cv.waitKey(1)
        # plt.show()
    return the_rect

def get_boxes_old(im):

    original_size = im.shape[:2]

    im = cv.resize(im, (500, 500*im.shape[0]//im.shape[1]))
    him, wim = im.shape[:2]

    # cv.imshow('painting:',im)
    # cv.waitKey(0)
    
    im = linear_stretch( im, descriptors.get_bgr_concat_hist(im))

    #gradients
    kernel = np.ones((7,7),np.float32)/49
    im = cv.filter2D(im,-1,kernel)
    # cv.imshow('blur:', im)

    laplacian = cv.Laplacian(im,cv.CV_64F)
    extrem = (abs(np.min(laplacian))+abs(np.max(laplacian)))//2
    tol = extrem*0.35

    lap = 255 * np.uint8(abs(laplacian[:, :, 0])>tol) # * np.uint8(abs(laplacian[:, :, 1])>tol) * np.uint8(abs(laplacian[:, :,2])>tol)
    # cv.imshow('lap:',lap)
    mask = np.zeros_like(lap)
    
    mask[int(him*0.05):-int(him*0.05),int(wim*0.2):-int(wim*0.2)] = lap[int(him*0.05):-int(him*0.05),int(wim*0.2):-int(wim*0.2)]
    # cv.imshow('corte:',mask)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (wim//5,4))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,6))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (wim//10,1))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # cv.imshow('morf:',mask)

    
    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    draw = mask.copy()
    rectangles=[]
    for blob in contours:
        x, y, w, h = cv.boundingRect(blob)
        fill = check_box_fill(mask[:,:,0], x, y, w, h)[2]
        if h>5 and w*h<wim*him*0.35 and h/w<0.9 and fill > 0.35:
            cx = x+w//2
            cy = y+h//2
            if wim*0.2>abs(wim//2-cx) and him*0.17<abs(him//2-cy):
                rectangles.append((x,y,w,h))
                cv.rectangle(draw, (x,y), (x+w,y+h), (0,0,255))

    # cv.imshow('rects',draw)
    # cv.waitKey(0)

    rectangles = sorted(rectangles, key=lambda x: x[1])
    aux = []
    if len(rectangles) == 1:
        aux = rectangles
    else:
        i = 0
        while i < len(rectangles)-1:
            if abs(rectangles[i][1]+rectangles[i][3]-rectangles[i+1][1])<15:
                y = rectangles[i][1] 
                x = rectangles[i][0] if rectangles[i][0] < rectangles[i+1][0] else rectangles[i+1][0]
                h = abs(y - rectangles[i+1][1]-rectangles[i+1][3])
                w = rectangles[i][2] if rectangles[i][2] > rectangles[i+1][2] else rectangles[i+1][2]
                aux.append((x,y,w,h))
                i+=1
            else:
                aux.append(rectangles[i])
            if i+1 == len(rectangles)-1:
                aux.append(rectangles[i+1])
            i+=1

    for rect in aux:
        cv.rectangle(draw, (rect[0],rect[1]), (rect[0]+rect[2],rect[3]+rect[1]), (0,255,0))



    if len(aux) < 1:
        return None

    the_rect = aux[0] 
    if len(aux)>1:
        the_rect = choose_best_rectangle_old(aux, laplacian)
    
    #expand_rect(aux[0])

    box_mask = np.zeros_like(mask[:,:,0])
    box_mask[the_rect[1]:the_rect[1]+the_rect[3], the_rect[0]:the_rect[0]+the_rect[2]] = np.ones((the_rect[3],the_rect[2]), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (wim//25,10))
    box_mask = cv.morphologyEx(box_mask, cv.MORPH_DILATE, kernel, iterations=2)

    # cv.imshow('result:',im[:,:,1]*box_mask)
    box_mask = cv.resize(box_mask, original_size[::-1])

    # cv.waitKey(0)
    return box_mask

def choose_best_rectangle_old(rects,laplacian):
    the_rect = None
    min_sumita = np.inf
    for rect in rects:
        again = laplacian[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        vector = again[again.shape[0]//2, :, :]
        cv.normalize(vector,vector,1,norm_type=cv.NORM_L2)
        # vector[:,0] = vector[:,0]/abs(np.max(vector[:,0]))
        # vector[:,1] = vector[:,1]/abs(np.max(vector[:,1]))
        # vector[:,2] = vector[:,2]/abs(np.max(vector[:,2]))
        sumita = np.sum(abs(vector[:,0]-vector[:,1]) + abs(vector[:,0]-vector[:,2]) + abs(vector[:,1]-vector[:,2]))/len(vector)
        if sumita < min_sumita:
            the_rect = rect
            min_sumita = sumita
        # print(sumita)
        # print(vector.shape)

        # cv.imshow('a ver:', again*255)
        # plt.plot(again[again.shape[0]//2, :])
        # cv.waitKey(1)
        # plt.show()
    return the_rect

def expand_rect(rects):
    #TODO:
    return None

def linear_stretch(im, hist_concat):
    if len(hist_concat) == 768:
        # Split concantenated hist
        hist = np.zeros((3, 256))
        hist[0] = np.transpose(hist_concat[:256])
        hist[1] = np.transpose(hist_concat[256:512])
        hist[2] = np.transpose(hist_concat[512:])
    elif len(hist_concat) == 512:
        # Split concantenated hist
        hist = np.zeros((2, 256))
        hist[0] = np.transpose(hist_concat[:256])
        hist[1] = np.transpose(hist_concat[256:512])
    else:
        hist = np.transpose(hist_concat)
    thresh = 0.0025

    for c in range(hist.shape[0]):
        ini = 0
        end = 255
        while(hist[c][ini] < thresh):
            ini += 1
        while(hist[c][end] < thresh):    
            end -= 1

        a = np.zeros_like(im[:, :, c], dtype=np.float64)
        a[:][:] = im[:, :, c]
        a = np.subtract(a, ini, out=np.zeros_like(a), where=((a - ini)>0), dtype=np.float64)
        if end == ini:
            end+=1
        a =  a / (end - ini)
        a = np.multiply(a, 255 , out=np.ones_like(a)*255, where=(a *255<255))
        im[:, :, c] = np.uint8(a[:][:])

    return im

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


# # reading images from queryset
# path = ['..', '..','datasets', 'qsd1_w3']
# qs_number = len([name for name in os.listdir(os.path.join(*path)) \
#     if '.jpg' in name])


# for i in range(qs_number):
#     path = ['..', '..','datasets', 'qsd1_w3', '{:05d}'.format(i)+'.jpg']
#     img= cv.imread(os.path.join(*path), cv.IMREAD_COLOR)
#     img = cv.resize(img, (500, 500*img.shape[0]//img.shape[1]))
#     get_boxes(img)
