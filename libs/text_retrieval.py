import pytesseract
import cv2
import numpy as np
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\users\juan1\AppData\Local\Tesseract-OCR\tesseract.exe"


# image = cv2.imread(r'../../datasets/qsd1_w3/00003.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def is_black_over_white(im):
    '''
    Checks whether the input (binarised) images is black over white or the other way around
    '''
    im_area = im.shape[0]*im.shape[1]
    num_white = np.sum(im/255)

    # print(100*num_white/im_area, '% white')
    return  100*num_white/im_area > 50

def draw_hough_line(im, theta, rho, color):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(im,(x1,y1),(x2,y2),color,2)



def morph_ocr(im):
    '''
    Applies morphology to improve ocr results
    '''
    # Remove small dots
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.array([1]))

    return im

def hough(im):
    cv2.imshow('hough', im)
    lines = cv2.HoughLines(im, 1, np.pi/180, 50, 0, None, 0, 0)

    horizontal_lines = []
    vertical_lines = []

    if lines is None:
        return
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for l in lines:
        for rho,theta in l:
            deg = round(180*theta/(np.pi))
            if deg == 0 or deg == 180: # vertical lines
                new = True
                for v in vertical_lines:
                    if abs(rho-v[1]) < 5:
                        new = False
                if new:
                    vertical_lines.append((theta, rho))
            elif deg == 90 or deg == 270: # hotizonal lines
                new = True
                for h in horizontal_lines:
                    if abs(rho-h[1]) < 0.1:
                        new = False
                if new:
                    horizontal_lines.append((theta, rho))

    for h in horizontal_lines:
        draw_hough_line(im, h[0], h[1], (255,0,0))
    for v in vertical_lines:
        draw_hough_line(im, v[0], v[1], (0, 255,0))

    cv2.imshow('hough', im)

def get_text(im):
    kernel = np.ones((5, 5), np.uint8)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',cv2.resize(im, (500, 500*im.shape[0]//im.shape[1])))
    # gray = cv2.morphologyEx(gray,cv2.MORPH_ERODE,kernel,dst=gray)
    gray = cv2.GaussianBlur(im,(3,3),0)
    # gray = cv2.medianBlur(gray, 3)
    # gray = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if not is_black_over_white(gray):
        gray = cv2.bitwise_not(gray)
    gray = morph_ocr(gray)
    # gray = cv2.Laplacian(gray, -1)
    # print(gray.shape)
    # cv2.imshow('tresh',gray)
    # cv2.imshow('img',im)
    # cv2.waitKey()
    # print('-'*20)
    custom_config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzñÑçÇ -c tessedit_char_blacklist=@ --oem 3'
    # abcdefgijklmnopqrstuvwxyzñç
    text = pytesseract.image_to_string(gray)#, config=custom_config)

    whitelist = set([c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzñÑçÇ/-'"])
    new_text = ''
    for c in text.strip():
        if c in whitelist or c == ' ':
            new_text += c
    
    
    # print(text)
    # print(len(text.strip()))
    print(new_text)
    # print('-'*20)
    # print(len(new_text))
    # cv2.imshow('gray',cv2.resize(gray, (500, 500*gray.shape[0]//gray.shape[1])))
    # cv2.waitKey()
    return new_text

# from libs import box_retrieval
# from libs import distance_metrics
# from libs import text_retrieval

# img = cv.imread(r'../datasets/qsd1_w3/00026.jpg')
# box = box_retrieval.get_boxes(img)
# cv.imshow('box',box*255)
# cv.waitKey()
# masked_img = cv.bitwise_and(img,img,mask = box)
# cv.imshow('masked_box',masked_img)
# cv.waitKey()
# text = text_retrieval.get_text(masked_img)
# print(text)