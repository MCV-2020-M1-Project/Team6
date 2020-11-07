import cv2
import glob
import os
import numpy as np


def BGR_2_LMS(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float64(img)
    img /= img.max() 
    conversion_matrix = np.array([[0.3811, 0.5783, 0.0402], \
                                    [0.967, 0.7244, 0.0782], \
                                    [0.0241, 0.1288, 0.8444]])
    LMS = np.array([conversion_matrix @ (img[i,j,:] ) for i in range(img.shape[0]) for j in range(img.shape[1]) ]).reshape(img.shape)
    LMS= np.log10(LMS+np.finfo(dtype=np.float64).eps)
    return LMS

def LMS_2_BGR(img):

    img = np.float64(img)
    img = 10**img
    img /= img.max() 
    conversion_matrix = np.array([[4.4679, -3.5873, 0.1193], \
                                    [-1.2186, 2.3809, -0.1624], \
                                    [0.0497, -0.2439, 1.2045]])
    RGB = np.array([conversion_matrix @ img[i,j,:] for i in range(img.shape[0]) for j in range(img.shape[1]) ]).reshape(img.shape)
    BGR = cv2.merge((RGB[:,:,2],RGB[:,:,1],RGB[:,:,0]))
    return BGR

def LMS_2_lalfabeta(img):
    img = np.float64(img)
    img /= img.max() 
    conversion_matrix = np.array([[1/np.sqrt(3), 0, 0], \
                                 [0, 1/np.sqrt(6), 0],  \
                                 [0, 0, 1/np.sqrt(2)]])
    conversion_matrix = conversion_matrix @ np.array([[1,1,1], \
                                                    [1,1,-2], \
                                                    [1,-1,0]])
    lalfabeta = np.array([conversion_matrix @ img[i,j,:]  for i in range(img.shape[0]) for j in range(img.shape[1]) ]).reshape(img.shape)
    return lalfabeta

def lalfabeta_2_LMS(img):
    img = np.float64(img)
    img /= img.max() 
    conversion_matrix = np.array([[1,1,1], \
                                  [1,1,-1], \
                                  [1,-2,0]])
    conversion_matrix = conversion_matrix @ np.array([[np.sqrt(3)/3, 0, 0], \
                                                    [0, np.sqrt(6)/6, 0],  \
                                                    [0, 0, np.sqrt(2)/2]])
    LMS = np.array([conversion_matrix @ (img[i,j,:] ) for i in range(img.shape[0]) for j in range(img.shape[1]) ]).reshape(img.shape)
    return LMS

path = ['..','datasets', 'qsd2_w2', '*.jpg']
input_files = glob.glob(os.path.join(*path))

# for index, image in enumerate(input_files):
im1 = cv2.imread(input_files[0])
im1 = cv2.resize(im1, (512,512*im1.shape[0]//im1.shape[1]))
im2 = cv2.imread(input_files[1])
im2 = cv2.resize(im2, (512,512*im2.shape[0]//im2.shape[1]))

lalfb_source = LMS_2_lalfabeta(BGR_2_LMS(im1))
lalfb_target = LMS_2_lalfabeta(BGR_2_LMS(im2))

l_source = lalfb_source[:,:,0] - np.mean(lalfb_source[:,:,0])
alpha_source = lalfb_source[:,:,1] - np.mean(lalfb_source[:,:,1])
beta_source = lalfb_source[:,:,2] - np.mean(lalfb_source[:,:,2])

# l_target = lalfb_target[0] - np.mean(lalfb_target[0])
# alpha_target = lalfb_target[1] - np.mean(lalfb_target[1])
# beta_target = lalfb_target[2] - np.mean(lalfb_target[2])

l = np.std(lalfb_target[:,:,0])/np.std(lalfb_source[:,:,0]) * l_source
alpha = np.std(lalfb_target[:,:,1])/np.std(lalfb_source[:,:,1]) * alpha_source
beta = np.std(lalfb_target[:,:,2])/np.std(lalfb_source[:,:,2]) * beta_source
# the resulting data points have standard deviations that conform to the target

# instead of adding the averages that we previously subtracted, we add the averages computed for the target
l = l + np.mean(lalfb_target[:,:,0])
alpha = alpha + np.mean(lalfb_target[:,:,1])
beta = beta + np.mean(lalfb_target[:,:,2])

lalfabeta_new = cv2.merge((l,alpha,beta))

new_img = LMS_2_BGR(lalfabeta_2_LMS(lalfabeta_new))

cv2.imshow('source', im1)
cv2.imshow('target', im2)
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
