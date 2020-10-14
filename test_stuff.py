import pickle as pkl
import cv2
import numpy as np
import descriptor_lib as descs
import distance_metrics_lib as dists
import os


def main():

    BD_path = ['..', 'datasets']

    BD_img = 'bbdd_00170.jpg'
    query_img = '00001.jpg'

    im1 = cv2.imread(os.path.join(*BD_path, 'BBDD', BD_img), 1)
    im2 = cv2.imread(os.path.join(*BD_path, 'qsd1_w1', query_img), 1)

    with open('gray_sim_mat.pkl', 'rb') as f:
        M = pkl.load(f)

    cv2.imshow('img1',cv2.resize(im1, (256, 256)))
    cv2.imshow('img2',cv2.resize(im2, (256, 256)))


    a = descs.get_descriptors(im1)
    b = descs.get_descriptors(im2)
    
    DESCR = 'gray_hist'

    a_bef = a[DESCR].copy()
    b_bef = b[DESCR].copy()
    norm_type = cv2.NORM_L2
    cv2.normalize(a_bef, a_bef, alpha=1, norm_type=norm_type)
    cv2.normalize(b_bef, b_bef, alpha=1, norm_type=norm_type)

    dif = a_bef - b_bef
    A = np.matmul(np.transpose(dif), M)
    dist = np.matmul(A, dif)
    print(dist)

    dists.display_comparison(a_bef, b_bef)


if __name__ == '__main__':
    main()