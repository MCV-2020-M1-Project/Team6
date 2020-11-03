import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp = sift.detect(gray,None)
# kp, des = sift.compute(gray,kp)
# kp, des = sift.detectAndCompute(gray,None)
# print(len(kp))
# img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.imwrite('sift_keypoints.jpg',img)

def get_SIFT_desc(img):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img,None)
    temp = dict([('des',des1)])
    # print(temp)
    return temp


def SIFT(img1,img2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    print(type(des2[0]))
    print('sift des2=',des2[0])
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.2*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    print(len(good))
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def ORB(img1,img2):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    print(len(matches))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()


def load_index():
    path = ['..','pkl_data','kp_bd_descriptors.pkl'] #for making the path system independent
    db_descript_list = []
    with open(os.path.join(*path), 'rb') as dbfile:
        db_descript_list = pkl.load(dbfile)
    return db_descript_list

def get_measures_SIFT(img1,mask,BFF=True):
    results = []
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1= sift.detectAndCompute(img1,mask=mask)

    db_list = load_index()
    print(db_list[2].values())

    for item in db_list:
        # print(item.keys())
        print('idex=',item['idx'])
        des2 = item['des']
        if des2 is None: continue
        # print(type(des2[0]))
        print('measures=',des2.shape)
        # quit()
        if BFF:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
        else:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        print(item['idx'] ,' , matching points= ', len(good))
        results.append((item['idx'],len(good)))
    return results

# img1 = cv2.imread(r'../../datasets/BBDD/bbdd_00104.jpg')
img2 = cv2.imread(r'../../datasets/qsd1_w4/00008.jpg', cv2.IMREAD_COLOR)
# ORB(img1,img2)
# SIFT(img1, img2)

# get_SIFT_desc(img2)

from libs import box_retrieval,background_removal
import retrieval_evaluation as re

mask_background = background_removal.method_canny_multiple_paintings(img2.copy())[1]

masks = re.sort_rects_lrtb(mask_background)
print(masks)
for mask in mask_background:
    if abs(mask[1] - mask[3]) < 50:
        continue

    v1 = mask[:2]
    v2 = mask[2:]
    img_no_bg = img2[v1[1]:v2[1], v1[0]:v2[0]]

mask = box_retrieval.get_boxes(img_no_bg)

results = get_measures_SIFT(img_no_bg, 1-mask, False)

results.sort(key=lambda x: x[1],reverse=True)
print(results[:3])