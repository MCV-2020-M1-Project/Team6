import pytesseract
import cv2
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# image = cv2.imread(r'../../datasets/qsd1_w3/00003.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_text(im):
    kernel = np.ones((5, 5), np.uint8)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',cv2.resize(im, (500, 500*im.shape[0]//im.shape[1])))
    # cv2.waitKey(0)
    # gray = cv2.morphologyEx(gray,cv2.MORPH_ERODE,kernel,dst=gray)
    gray = cv2.GaussianBlur(im,(3,3),0)
    # gray = cv2.medianBlur(gray, 3)
    # gray = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('tresh',gray)
    # cv2.imshow('img',im)
    cv2.waitKey()
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
    # print(new_text)
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