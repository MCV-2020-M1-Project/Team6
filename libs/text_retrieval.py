import pytesseract
import cv2
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# image = cv2.imread(r'../../datasets/qsd1_w3/00003.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

from libs import box_retrieval
from libs import distance_metrics
from libs import text_retrieval
img = cv.imread(r'../datasets/qsd1_w3/00026.jpg')

box = box_retrieval.get_boxes(img)
cv.imshow('box',box*255)
cv.waitKey()
masked_img = cv.bitwise_and(img,img,mask = box)
cv.imshow('masked_box',masked_img)
cv.waitKey()
text = text_retrieval.get_text(masked_img)
print(text)

def get_text(im):
    kernel = np.ones((5, 5), np.uint8)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # gray = cv2.morphologyEx(gray,cv2.MORPH_ERODE,kernel,dst=gray)
    gray = cv2.GaussianBlur(im,(3,3),0)
    # gray = cv2.medianBlur(gray, 3)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(gray)

    # print(text)
    # cv2.imshow('gray',gray)
    # cv2.waitKey()
    return text

