import cv2 as cv
import numpy as np


def mask_evaluation(annotation, result):
    #result=cv.imread(r"sandbox\\mask-dilate.png",0)
    #annotation=cv.imread(r"qsd2_w1\\00004.png",0)

    result_neg = cv.bitwise_not(result)
    annotation_neg = cv.bitwise_not(annotation)

    tp = annotation & result
    fn = annotation & result_neg
    fp = annotation_neg & result
    tn = annotation_neg & result_neg

    cv.imshow('TP', tp)
    cv.imshow('FN', fn)
    cv.imshow('FP', fp)
    cv.imshow('TN', tn)

    cv.waitKey()

    tp = tp.sum()
    fn = fn.sum()
    fp = fp.sum()
    tn = tn.sum()

    print(tp, fn, fp, tn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_measure = 2 * (precision * recall / (precision + recall))

    print("Precision=", precision)
    print("Recall=", recall)
    print("F1 measure=", f1_measure)

    return precision, recall, f1_measure
