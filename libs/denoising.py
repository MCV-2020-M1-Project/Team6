'''
Library for all related to denoising
'''
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def get_ssim(original, denoised):
    return ssim(original, denoised, data_range=denoised.max() - denoised.min(), multichannel=True)


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
    methods = {}
    # normal
    methods['normal'] = cv2.blur(img, (5,5)) 

    # Gaussian
    methods['gauss'] = cv2.GaussianBlur(img,(5,5),0)

    # Median
    methods['median'] = cv2.medianBlur(img, 5)

    # Bilateral
    # methods['bilateral'] = cv2.bilateralFilter(img, 5, 75, 75)

    psnr_noise_eval = sorted([(get_psnr(img, m), k) for k, m in methods.items()], key=lambda x: x[0], reverse=True)
    mad_noise_eval = sorted([(get_mad(img, m), k) for k, m in methods.items()], key=lambda x: x[0])
    ssim_noise_eval = sorted([(get_ssim(img, m), k) for k, m in methods.items()], key=lambda x: x[0], reverse=True)

    print('psnr:', psnr_noise_eval, abs(psnr_noise_eval[0][0] -  psnr_noise_eval[1][0]))
    print('mad:', mad_noise_eval, abs(mad_noise_eval[0][0] -  mad_noise_eval[1][0]))
    print('ssim:', ssim_noise_eval)

    '''
    Notes:
    mad seems useful to decide if the image is noisy at all or not. Values sammler than 15 (or maybe 10) generally coincide with non-noisy images
    Gaussian noise gets the best psnr practically eveytime, but when median is not that worse (maybe an absolute difference of 1.5) median probably
    is better for colour based tasks (but it tends to destroy letters, so there gaussian (or nothing) is always prefferable). So, the algorithm for method 1 
    would be smth like:

    if mad > 15:
        if abs(gass - median) < 15:
            color_based = median
            ocr = gaussian
        else:
            color_based = ocr = gaussian
    '''
    if display:
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

        cv2.waitKey(0)


for i in range(30):
    print(i)
    img_path = '../datasets/qsd1_w3/{:05d}.jpg'.format(i)
    img = cv2.imread(img_path)
    img_gray = cv2.imread(img_path, 0)
    img_v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:, 2]

    print('rgb')
    find_best_denoise(img, True)
