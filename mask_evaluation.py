import cv2 as cv
import numpy as np

def mask_evaluation(annotation, result):
    #result=cv.imread("mask-dilate.png",0)
    #annotation=cv.imread(r"C:\Users\adama\OneDrive\Dokumenty\Master of Computer Vision\M1\week 1\qsd2_w1\00004.png",0)

    result_neg=cv.bitwise_not(result)
    annotation_neg=cv.bitwise_not(annotation)

    TP = annotation & result
    FN = annotation & result_neg
    FP = annotation_neg & result
    TN = annotation_neg & result_neg

    cv.imshow('TP',TP)
    cv.imshow('FN',FN)
    cv.imshow('FP',FP)
    cv.imshow('TN',TN)

    cv.waitKey()

    TP=TP.sum()
    FN=FN.sum()
    FP=FP.sum()
    TN=TN.sum()

    print(TP,FN,FP,TN)

    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1_measure=2*(precision*recall/(precision+recall))

    print("Precision=",precision)
    print("Recall=",recall)
    print("F1 measure=",f1_measure)

    return (precision,recall,f1_measure)

