import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from libs import box_retrieval,background_removal,distance_metrics
import retrieval_evaluation as re
from skimage.feature import daisy

# img = cv2.imread('../datasets/qsd1_w4/00000.jpg')
# if img is None:
#     print('Error reading image')
#     quit()
# gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp = sift.detect(gray,None)
# kp, des = sift.compute(gray,kp)
# kp, des = sift.detectAndCompute(gray,None)
# print(len(kp))
# img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.imwrite('sift_keypoints.jpg',img)

"""
That's the file where you can quickly evaluate picture by picture and draw them
works along with index_db_kp_desc
"""

def get_SIFT_desc(img, mask=None):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img,mask)
    temp = des1
    # print(temp)
    return temp

def get_ORB_desc(img, mask=None):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, mask)
    # compute the descriptors with ORB
    kp, des1 = orb.compute(img, kp)
    temp = des1
    return temp

def SIFT(img1,img2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if len(kp1) < 2 or len(kp2) < 2:
        return 0
    # BFMatcher with default params
    # print(type(des2[0]))
    # print('sift des2=',des2[0])
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.3*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    # # print('Num matches:',  len(good))
    # # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # # plt.imshow(img3),plt.show()
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # good = [m for m in matches if m.distance < 55]
    return len(good)

def ORB(img1,img2):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    print(len(kp1))
    print(kp1[0].pt, kp1[0].angle)
    good = []
    if True: # method 1 for getting matches
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        # print(len(matches))
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3), plt.show()
        good = [m for m in matches if m.distance < 21]
        # cv.drawMatchesKnn expects list of lists as matches.
        # print('Num matches:',  len(good))
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append([m])

    return len(good)


def load_index():
    path = ['..','pkl_data','kp_bd_descriptors.pkl'] #for making the path system independent
    db_descript_list = []
    with open(os.path.join(*path), 'rb') as dbfile:
        db_descript_list = pkl.load(dbfile)
    return db_descript_list

def get_bf_matching(des1, des2,descriptor=None):
    if descriptor == 'orb2':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
    return matches

def get_flann_matching(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def get_best_matches(des1,descriptor, measure = 'BFM'):
    results = []
    db_list = load_index()

    for item in db_list:
        # print(item.keys())
        print('idex=',item['idx'])
        print("desc=",descriptor)
        des2 = item[descriptor]
        if des2 is None: continue
        # print(type(des2[0]))
        print('shape of des=',des2.shape)
        # quit()
        if measure == 'BFM':
            matches = get_bf_matching(des1,des2, descriptor)
        elif measure == 'flann':
            matches = get_flann_matching(des1, des2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        print(item['idx'] ,' , matching points= ', len(good))
        results.append((item['idx'],len(good)))
    return results

def compare_SIFT(img1,descriptor='sift',mask=None,measure='BFM'):
    # Initiate SIFT detector
    des1 = get_SIFT_desc(img1,mask)
    db_list = load_index()
    # print(db_list[2].values())

    return get_best_matches(des1,descriptor, measure)

def compare_ORB(img1,descriptor='orb',mask=None,measure='BFM'):
    # Initiate ORB detector
    des1 = get_ORB_desc(img1,mask)

    db_list = load_index()
    # print(db_list[2].values())

    return get_best_matches(des1,descriptor,measure)

def DAISY(img1, img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (512, 512), img1)
    img2 = cv2.resize(img2, (512, 512), img2)

    des1,viz1= daisy(img1, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)
    len(des1)

    des2,viz2= daisy(img2, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)
    # BFMatcher with default params
    # result = get_flann_matching(des1[0],des2[0])

    # cv.drawMatchesKnn expects list of lists as matches.
    # print(len(result))

    vectors = min(len(des1), len(des2))
    hist1 = np.concatenate([des1[x][y] for x in range(0, vectors) for y in range(0, vectors)])
    hist2 = np.concatenate([des2[x][y] for x in range(0, vectors) for y in range(0, vectors)])
    outcome =distance_metrics.get_l1_distance(hist1,hist2)

    print('hist=',len(hist1))

    result = outcome


    return viz1,viz2,result


def main():
    # img1 = cv2.imread(r'../../datasets/BBDD/bbdd_00104.jpg')
    # img2 = cv2.imread(r'../../datasets/qsd1_w4/00008.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread(r'../datasets/BBDD/bbdd_00104.jpg')
    test = 0
    img2 = cv2.imread('../datasets/qsd1_w4/{:05d}.jpg'.format(test), cv2.IMREAD_COLOR)
    # 250,6
    # ORB(img1,img2)
    # SIFT(img1, img2)
    # get_SIFT_desc(img2)

    mask_background = background_removal.method_canny_multiple_paintings(img2.copy())[1]

    masks = re.sort_rects_lrtb(mask_background)
    # print(masks)
    for mask in masks:
        if abs(mask[1] - mask[3]) < 50:
            continue

        v1 = mask[:2]
        v2 = mask[2:]
        img_no_bg = img2[v1[1]:v2[1], v1[0]:v2[0]]
    # cv2.imshow('im', img_no_bg)
    # cv2.waitKey(0)
    mask = box_retrieval.get_boxes(img_no_bg)
    # des= DAISY(img_no_bg)
    img1=cv2.resize(img1,(512,512),img1)
    img_no_bg = cv2.resize(img_no_bg, (512, 512), img_no_bg)


    print(f'testing for image {test}:')
    for i in range(287):
        img1 = cv2.imread('../datasets/BBDD/bbdd_{:05d}.jpg'.format(i))
        img1=cv2.resize(img1,(512,512),img1)    
        m = SIFT(img1,img_no_bg)
        # m = ORB(img1,img_no_bg)
        if m > 0:
            print(i, m)

    # ORB(img1,img_no_bg)
    print('Done')
    quit()
    viz1,viz2, result = DAISY(img1,img_no_bg)

    viz1=cv2.resize(viz1, (512,512),viz1)
    viz2=cv2.resize(viz2, (512, 512), viz2)
    print('shape1',viz1.shape)
    print('shape2', viz2.shape)
    print('result=',result)
    cv2.imshow('viz1', viz1)
    cv2.imshow('viz2',viz2)
    cv2.waitKey()
    quit()
    # results = compare_SIFT(img_no_bg, descriptor='sift',mask= 1-mask, measure='flann')
    print('surf')
    results = compare_ORB(img_no_bg,descriptor='orb', mask=1-mask, measure='BFM')

    results.sort(key=lambda x: x[1],reverse=True)
    print(results[:3])

main()

# hist = np.concatenate(des1[x][y] for x in range(0, vectors))