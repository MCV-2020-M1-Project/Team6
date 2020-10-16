import cv2
import numpy as np


def linear_stretch(im, hist_concat):
    if len(hist_concat) == 768:
        # Split concantenated hist
        hist = np.zeros((3, 256))
        hist[0] = np.transpose(hist_concat[:256])
        hist[1] = np.transpose(hist_concat[256:512])
        hist[2] = np.transpose(hist_concat[512:])
    elif len(hist_concat) == 512:
        # Split concantenated hist
        hist = np.zeros((2, 256))
        hist[0] = np.transpose(hist_concat[:256])
        hist[1] = np.transpose(hist_concat[256:512])
    else:
        hist = np.transpose(hist_concat)

    # print('Descr size', hist.shape)

    num_pixels = im.shape[0] * im.shape[1]
    thresh = 0.00025 * num_pixels

    for c in range(hist.shape[0]):
        ini = 0
        end = 255
        # print(hist[c][ini], 'ini = ', ini)
        while(hist[c][ini] < thresh):
            # print(hist[c][ini], 'ini = ', ini)
            ini += 1
        
        # print(hist[c][end], 'end = ', end)
        while(hist[c][end] < thresh):
            # print(hist[c][end], 'end = ', end)    
            end -= 1

        a = np.zeros_like(im[:, :, c], dtype=np.float64)
        a[:][:] = im[:, :, c]
        a = np.subtract(a, ini, out=np.zeros_like(a), where=((a - ini)>0), dtype=np.float64)
        if end == ini:
            end+=1
        a =  a / (end - ini)
        a = np.multiply(a, 255 , out=np.ones_like(a)*255, where=(a *255<255))
        im[:, :, c] = np.uint8(a[:][:])

    return im

def get_gray_hist(img, mask=None):
    '''Paramenters: img (color image)
        Returns: hist (grayscale histogram) '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], mask, [256], [0,256])

    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_bgr_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 bgr histograms concatenated '''

    hist_b = cv2.calcHist([img], [0], mask, [256], [0,256])
    hist_g = cv2.calcHist([img], [1], mask, [256], [0,256])
    hist_r = cv2.calcHist([img], [2], mask, [256], [0,256])

    hist = np.concatenate((hist_b, hist_g, hist_r))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_cielab_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 lab histograms concatenated '''

    cielab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    hist_l = cv2.calcHist([cielab], [0], mask, [256], [0,256])
    hist_a = cv2.calcHist([cielab], [1], mask, [256], [0,256])
    hist_b = cv2.calcHist([cielab], [2], mask, [256], [0,256])

    hist = np.concatenate((hist_l, hist_a, hist_b))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_ycrcb_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 YCrCb histograms concatenated '''

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hist_y = cv2.calcHist([ycrcb], [0], mask, [256], [0,256])
    hist_cr = cv2.calcHist([ycrcb], [1], mask, [256], [0,256])
    hist_cb = cv2.calcHist([ycrcb], [2], mask, [256], [0,256])

    hist = np.concatenate((hist_y, hist_cr, hist_cb))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_hsv_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 HSV histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0,256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0,256])
    hist_v = cv2.calcHist([hsv], [2], mask, [256], [0,256])    

    hist = np.concatenate((hist_h, hist_s, hist_v))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_hs_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 2 HS histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0,256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0,256])

    hist = np.concatenate((hist_h, hist_s))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_hs_concat_hist_blur(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 2 HS histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0,256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0,256])

    hist = np.concatenate((hist_h, hist_s))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_hsv_concat_hist_blur(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 HSV histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    hist_h = cv2.calcHist([hsv],[0], mask, [256], [0,256])
    hist_s = cv2.calcHist([hsv],[1], mask, [256], [0,256])
    hist_v = cv2.calcHist([hsv],[2], mask, [256], [0,256])

    hist = np.concatenate((hist_h, hist_s, hist_v))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_h_multi_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: return several hist concat result of splitting the image '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    (h, w) = hsv.shape[:2]
    tile1 = hsv[0:h//2, 0:w//2]
    tile2 = hsv[0:h//2, w//2:-1]
    tile3 = hsv[h//2:-1, 0:w//2]
    tile4 = hsv[h//2:-1, w//2:-1]

    mask1 = mask2 = mask3 = mask4 = None
    if mask is not None:
        mask1 = mask[0:h//2, 0:w//2]
        mask2 = mask[0:h//2, w//2:-1]
        mask3 = mask[h//2:-1, 0:w//2]
        mask4 = mask[h//2:-1, w//2:-1]


    hist_tile1 = cv2.calcHist([tile1],[0],mask1,[256],[0,256])
    hist_tile2 = cv2.calcHist([tile2],[0],mask2,[256],[0,256])
    hist_tile3 = cv2.calcHist([tile3],[0],mask3,[256],[0,256])
    hist_tile4 = cv2.calcHist([tile4],[0],mask4,[256],[0,256])

    hist = np.concatenate((hist_tile1, hist_tile2, hist_tile3, hist_tile4))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_hs_multi_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: return several hist concat result of splitting the image '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    (h, w) = hsv.shape[:2]
    tile1 = hsv[0:h//2, 0:w//2]
    tile2 = hsv[0:h//2, w//2:-1]
    tile3 = hsv[h//2:-1, 0:w//2]
    tile4 = hsv[h//2:-1, w//2:-1]

    mask1 = mask2 = mask3 = mask4 = None
    if mask is not None:
        mask1 = mask[0:h//2, 0:w//2]
        mask2 = mask[0:h//2, w//2:-1]
        mask3 = mask[h//2:-1, 0:w//2]
        mask4 = mask[h//2:-1, w//2:-1]

    hist_h_tile1 = cv2.calcHist([tile1],[0],mask1,[256],[0,256])
    hist_h_tile2 = cv2.calcHist([tile2],[0],mask2,[256],[0,256])
    hist_h_tile3 = cv2.calcHist([tile3],[0],mask3,[256],[0,256])
    hist_h_tile4 = cv2.calcHist([tile4],[0],mask4,[256],[0,256])

    hist_s_tile1 = cv2.calcHist([tile1],[1],mask1,[256],[0,256])
    hist_s_tile2 = cv2.calcHist([tile2],[1],mask2,[256],[0,256])
    hist_s_tile3 = cv2.calcHist([tile3],[1],mask3,[256],[0,256])
    hist_s_tile4 = cv2.calcHist([tile4],[1],mask4,[256],[0,256])

    hist = np.concatenate((hist_h_tile1, hist_h_tile2, hist_h_tile3, hist_h_tile4, \
        hist_s_tile1, hist_s_tile2, hist_s_tile3, hist_s_tile4))

    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_hs_concat_hist_st(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 HSV histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0,256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0,256])
    hist_concat = np.concatenate((hist_h, hist_s))
    hsv_st = linear_stretch(hsv, hist_concat)
    hist_h = cv2.calcHist([hsv_st], [0], mask, [256], [0,256])
    hist_s = cv2.calcHist([hsv_st], [1], mask, [256], [0,256])

    hist = np.concatenate((hist_h, hist_s))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist

def get_descriptors(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: descript_dic (dictionary with descriptors names as keys) '''

    descript_dic = {}
    # descript_dic['gray_hist'] = get_gray_hist(img, mask)
    # descript_dic['bgr_concat_hist'] = get_bgr_concat_hist(img, mask)
    # descript_dic['cielab_concat_hist'] = get_bgr_concat_hist(img, mask)
    # descript_dic['ycrcb_concat_hist'] = get_ycrcb_concat_hist(img, mask)
    descript_dic['hsv_concat_hist'] = get_hsv_concat_hist(img, mask)
    descript_dic['hs_concat_hist'] = get_hs_concat_hist(img, mask)
    # descript_dic['hs_concat_hist_st'] = get_hs_concat_hist_st(img, mask)
    # descript_dic['hs_concat_hist_blur'] = get_hs_concat_hist_blur(img, mask)
    # descript_dic['hsv_concat_hist_blur'] = get_hsv_concat_hist_blur(img, mask)
    # descript_dic['h_multi_hist'] = get_h_multi_hist(img, mask)
    descript_dic['hs_multi_hist'] = get_hs_multi_hist(img, mask)
    return descript_dic

