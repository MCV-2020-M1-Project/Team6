import pytesseract
import cv2
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread(r'../../datasets/qsd1_w3/00003.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
# gray = cv2.morphologyEx(gray,cv2.MORPH_ERODE,kernel,dst=gray)
gray = cv2.GaussianBlur(gray,(3,3),0)
# gray = cv2.medianBlur(gray, 3)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

text = pytesseract.image_to_string(gray)

print(text)
cv2.imshow('gray',gray)
cv2.waitKey()

