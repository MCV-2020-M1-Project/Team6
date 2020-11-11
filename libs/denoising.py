'''
Library for all related to denoising
'''
import os
import sys
sys.path.append(os.path.split(__file__)[0])

import numpy as np
import cv2

import background_removal as br
# from skimage.metrics import structural_similarity as ssim


def denoise_img(img):
    '''
    Gets a possibly noisy image and returns:
    if no need to denoise: img as is
    else: a dicr with keys 'ocr', 'color'
    '''
    
    k_size = np.max(img.shape)//300
    k_size = k_size if k_size % 2 else k_size + 1
    k_size = 3 if k_size < 3 else k_size

    gauss = cv2.GaussianBlur(img,(k_size, k_size), 0)
    median = cv2.medianBlur(img, k_size)

    gauss_psnr = get_psnr(img, gauss)
    gauss_mad = get_mad(img, gauss)
    median_psnr = get_psnr(img, median)

    denoised_img = img
    if gauss_mad > 20: 
        return cv2.medianBlur(img, 5) # hotfix mimicking other teams. Results actually increased  alittle bit

        if gauss_psnr > median_psnr and abs(get_psnr(img, gauss) - get_psnr(img, median)) > 2:
            denoised_img = gauss
        else:
            denoised_img = median
    return denoised_img


# def get_ssim(original, denoised):
#     return ssim(original, denoised, data_range=denoised.max() - denoised.min(), multichannel=True)


def get_mse(original, denoised):
    '''
    Computes Mean Square Error (the smaller the better)
    '''
    original, denoised = np.int16(original) , np.int16(denoised)
    error = original - denoised

    return np.sum(error*error) / error.shape[0] / error.shape[1]


def get_mad(original, denoised):
    '''
    Computes Mean Absolute Difference (the smaller the better)
    '''
    original, denoised = np.int16(original) , np.int16(denoised)
    error = original - denoised

    return  np.sum(np.abs(error)) / error.shape[0] / error.shape[1]


def get_psnr(original, denoised, max_value=255):
    '''
    Computes Peak Signal to Noise Ratio (the bigger the better)
    '''
    return 10*np.log10(max_value*max_value/get_mse(original, denoised))


def find_best_denoise(img, display=False):
    k_size = 3
    methods = {}
    # normal
    methods['normal'] = cv2.blur(img, (k_size,k_size)) 

    # Gaussian
    methods['gauss'] = cv2.GaussianBlur(img,(k_size,k_size),0)

    # Median
    methods['median'] = cv2.medianBlur(img, k_size)

    # Bilateral
    # methods['bilateral'] = cv2.bilateralFilter(img, 5, 75, 75)

    psnr_noise_eval = sorted([(get_psnr(img, m), k) for k, m in methods.items()], key=lambda x: x[0], reverse=True)
    mad_noise_eval = sorted([(get_mad(img, m), k) for k, m in methods.items()], key=lambda x: x[0])

    print('psnr:', psnr_noise_eval, abs(psnr_noise_eval[0][0] -  psnr_noise_eval[1][0]))
    print('mad:', mad_noise_eval, abs(mad_noise_eval[0][0] -  mad_noise_eval[1][0]))

    if display:
        while cv2.waitKey(0) != ord('q'):
            win_width = 400

            cv2.imshow('Original', img)
            for i in psnr_noise_eval:
                cv2.namedWindow(i[1])

            for i in psnr_noise_eval:
                s = methods[i[1]].shape
                cv2.imshow(i[1], cv2.resize(methods[i[1]], (win_width, win_width*s[0]//s[1])))
            
            xd = 10
            cv2.moveWindow('Original', xd, 10)
            xd += 20
            for i in psnr_noise_eval:
                xd += win_width + 10
                cv2.moveWindow(i[1], xd, 10)

        


def image_has_noise(image):
    # segment background
    painting_mask, _ = br.method_canny_multiple_paintings(image)
    image[:, :, 0] *= (1 - painting_mask)
    image[:, :, 1] *= (1 - painting_mask)
    image[:, :, 2] *= (1 - painting_mask)

    # decide using psnr
    blurred_background = cv2.medianBlur(image, 3)
    if get_psnr(image, blurred_background) > 30:
        return False
    return True

# for i in range(30):
#     print(i)
#     img_path = '../datasets/qsd1_w3/{:05d}.jpg'.format(i)
#     img = cv2.imread(img_path)
#     # img_gray = cv2.imread(img_path, 0)
#     # img_v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:, 2]

#     # print('rgb')
#     find_best_denoise(img, True)
# #     imgs = denoise_img(img)