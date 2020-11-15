import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog,daisy
import pytesseract

def painting_in_db(des1, dataset, mask=None, method=1):
    '''
    Method 1, 2 and 3 > Get only distance, differences are:
        - 1: Thresh is 21
        - 2: Thresh is 35
        - 3: Thresh us 35 and singles point matches are ignored
    Method 4 > Get 2 nearest matches and compared them. The first one must be significantly better than the second for it to be valid.
    Then, matches with 3 or less points are ignored
    '''
    total_sum = 150000

    des1 = des1['orb']

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    biggest_match = 0
    # Iterate over the ds
    for id,ds_im in enumerate(dataset):

        des2 = ds_im['orb']
        if des2 is None:
            continue
        # print(len(des1), len(des2))
        good = []
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        index_compare = len(matches)//10
        good = [m for m in matches if m.distance < 40]
        matches_calc = [m.distance for m in matches[:index_compare]]
        var = np.var(np.array(matches_calc))
        # print("id=",id,"len good = ", len(good) , "var=", var)
        # print('var=', var)
        # if len(good) > 4: return False
        if len(good)>biggest_match: biggest_match=len(good)
        if len(good) > 3:
            if var < 20:
                return False

    return True

def painting_in_db_old(des1, dataset, mask=None, method=1):
    '''
    Method 1, 2 and 3 > Get only distance, differences are:
        - 1: Thresh is 21
        - 2: Thresh is 35 
        - 3: Thresh us 35 and singles point matches are ignored
    Method 4 > Get 2 nearest matches and compared them. The first one must be significantly better than the second for it to be valid.
    Then, matches with 3 or less points are ignored
    '''
    total_sum = 0

    des1 = des1['orb']

    # Create matcher
    if method == 4:
        bf = cv2.BFMatcher()
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Iterate over the ds
    for ds_im in dataset:
        if total_sum > 0: 
            return True

        des2 = ds_im['orb']
        if des2 is None:
            continue
        # print(len(des1), len(des2))
        good = []
        if method == 4:
            matches = bf.knnMatch(des1,des2,k=2)
            for m,n in matches:
                if m.distance < 0.6*n.distance:
                    good.append([m])
            total_sum += len(good) if len(good) > 3 else 0
        else:
            # Match descriptors.
            matches = bf.match(des1, des2)

            if method == 1:
                good = [m for m in matches if m.distance < 21]
                total_sum += len(good) 
            else:
                good = [m for m in matches if m.distance < 35]
                if method == 3:
                    total_sum += len(good) if len(good) > 4 else 0
                else:
                    total_sum += len(good)
    return False


def painting_in_db2(des1, dataset, mask=None, method=1):
    '''
    Method 1, 2 and 3 > Get only distance, differences are:
        - 1: Thresh is 21
        - 2: Thresh is 35 
        - 3: Thresh us 35 and singles point matches are ignored
    Method 4 > Get 2 nearest matches and compared them. The first one must be significantly better than the second for it to be valid.
    Then, matches with 3 or less points are ignored
    '''
    total_sum = 0

    des1_sift = des1['sift']
    des1_orb = des1['orb']

    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf_sift = cv2.BFMatcher()


    # Iterate over the ds
    for ds_im in dataset:
        if total_sum > 0: 
            return True

        des2_sift = ds_im['sift']
        des2_orb = ds_im['orb']
        if des2_sift is None or des2_orb is None:
            continue
        # print(len(des1), len(des2))
        good = []

        # SIFT
        matches_sift = bf_sift.knnMatch(des1_sift,des2_sift,k=2)
        for m,n in matches_sift:
            if m.distance < 0.3*n.distance:
                good.append([m])
        total_sum += len(good) if len(good) > 0 else 0


        # ORB
        matches_orb = bf_orb.match(des1_orb, des2_orb)

        good = [m for m in matches_orb if m.distance < 21]
        total_sum += len(good) if len(good) > 0 else 0

    return False


def get_daisy_desc(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512, 512), img)
    des1= daisy(img, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=False)
    vectors = len(des1)
    hist1 = np.concatenate([des1[x][y] for x in range(0, vectors) for y in range(0, vectors)])
    cv2.normalize(hist1, hist1, norm_type=cv2.NORM_L2, alpha=1.)
    return hist1

def get_sift_desc(img, mask=None):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, mask)
    # temp = dict([('des',des1)])
    # print(temp)
    return des1


def get_orb_desc(img, mask=None):
    if mask is not None: mask = cv2.resize(mask,(512,512))
    img = cv2.resize(img,(512,512))
    # Initiate ORB detector
    orb = cv2.ORB_create(scaleFactor=1.1,fastThreshold=10,WTA_K=2)
    # find the keypoints with ORB
    # kp = orb.detect(img, mask)
    # # compute the descriptors with ORB
    # kp, des1 = orb.compute(img, kp)
    _, des1 = orb.detectAndCompute(img, mask)

    return des1


def get_hog(im):
    resized_im = cv2.resize(im, (512, 512))
    H = hog(resized_im, orientations=9, pixels_per_cell=(15, 15), cells_per_block=(2, 2),
            # TODO try smaller pixels per cell
            transform_sqrt=True, block_norm="L1")
    return H


def get_dct(img, N=100):
    imf = np.float32(img) / 255.0
    freq = cv2.dct(imf)
    ni, nj = freq.shape
    subset = freq[:N, :N]

    return subset.flatten()


def get_lbp(img, radius=2., block_w=8, n_points=None, method='uniform'):
    """
    dividing image to shape//block_w blocks
    calculating LBP for each block creating histogram out of LBP output
    concatinating histograms
    """
    n_points = 8 * radius if n_points is None else n_points

    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = img.shape

    h_blocks = shape[1] // block_w
    w_blocks = shape[0] // block_w
    list_hist = []

    for number_w in range(0, w_blocks):
        for number_h in range(0, h_blocks):
            if (number_w + 1) * block_w > shape[0]:
                x_end = shape[0]
            else:
                x_end = (number_w + 1) * block_w
            if (number_h + 1) * block_w > shape[1]:
                y_end = shape[1]
            else:
                y_end = (number_h + 1) * block_w
            x_front = number_w * block_w
            y_front = number_h * block_w
            img_block = img[x_front:x_end, y_front:y_end].copy()

            lbim = local_binary_pattern(img_block, n_points, radius, method)
            (hist, _) = np.histogram(lbim.ravel(), bins=np.arange(0,n_points+3),range=(0,n_points+2))
            list_hist.append(hist)


    full_hist = np.concatenate((list_hist))
            # result =+ np.uint8(255 * (lbim - lbim.min()) / (lbim.max() - lbim.min()))

    return full_hist


def get_DCT_coefs(image, N, block_w=8):
    """
    Function that get first N coeficients from DCT of 8x8 block
    from the image
    param:
    img = image from which you want to calculate DCT
    N = number of first coefficients you wish to extract
    """

    def get_zig_zag(img_block, N):
        matrix = img_block
        shape = matrix.shape
        # print(shape)
        rows = shape[0]
        columns = shape[1]
        coefs_zigzag = []

        solution = [[] for i in range(rows + columns - 1)]

        for i in range(rows):
            for j in range(columns):
                sum = i + j
                if (sum % 2 == 0):

                    # add at beginning
                    solution[sum].insert(0, matrix[i][j])
                else:

                    # add at end of the list
                    solution[sum].append(matrix[i][j])

                # print the solution as it as
        for i in solution:
            for j in i:
                # print(j, end=" ")
                coefs_zigzag.append(j)
        return coefs_zigzag[:N]

    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512, 512))
    shape = img.shape
    all_coefs = []

    h_blocks = shape[1] // block_w
    w_blocks = shape[0] // block_w

    for number_w in range(0, w_blocks):
        for number_h in range(0, h_blocks):
            if (number_w + 1) * block_w > shape[0]:
                x_end = shape[0]
            else:
                x_end = (number_w + 1) * block_w
            if (number_h + 1) * block_w > shape[1]:
                y_end = shape[1]
            else:
                y_end = (number_h + 1) * block_w
            x_front = number_w * block_w
            y_front = number_h * block_w
            img_block = img[x_front:x_end, y_front:y_end].copy()
            coefs = cv2.dct(img_block.astype(np.float32))
            feature_vector = get_zig_zag(coefs, N)
            all_coefs.extend(feature_vector)

    return np.array(all_coefs)


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
        while (hist[c][ini] < thresh):
            # print(hist[c][ini], 'ini = ', ini)
            ini += 1

        # print(hist[c][end], 'end = ', end)
        while (hist[c][end] < thresh):
            # print(hist[c][end], 'end = ', end)    
            end -= 1

        a = np.zeros_like(im[:, :, c], dtype=np.float64)
        a[:][:] = im[:, :, c]
        a = np.subtract(a, ini, out=np.zeros_like(a), where=((a - ini) > 0), dtype=np.float64)
        if end == ini:
            end += 1
        a = a / (end - ini)
        a = np.multiply(a, 255, out=np.ones_like(a) * 255, where=(a * 255 < 255))
        im[:, :, c] = np.uint8(a[:][:])

    return im


def get_gray_hist(img, mask=None):
    '''Paramenters: img (color image)
        Returns: hist (grayscale histogram) '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])

    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_bgr_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 bgr histograms concatenated '''

    hist_b = cv2.calcHist([img], [0], mask, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], mask, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], mask, [256], [0, 256])

    hist = np.concatenate((hist_b, hist_g, hist_r))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_cielab_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 lab histograms concatenated '''

    cielab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    hist_l = cv2.calcHist([cielab], [0], mask, [256], [0, 256])
    hist_a = cv2.calcHist([cielab], [1], mask, [256], [0, 256])
    hist_b = cv2.calcHist([cielab], [2], mask, [256], [0, 256])

    hist = np.concatenate((hist_l, hist_a, hist_b))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_ycrcb_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 YCrCb histograms concatenated '''

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hist_y = cv2.calcHist([ycrcb], [0], mask, [256], [0, 256])
    hist_cr = cv2.calcHist([ycrcb], [1], mask, [256], [0, 256])
    hist_cb = cv2.calcHist([ycrcb], [2], mask, [256], [0, 256])

    hist = np.concatenate((hist_y, hist_cr, hist_cb))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_hsv_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 HSV histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [256], [0, 256])

    hist = np.concatenate((hist_h, hist_s, hist_v))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_hs_concat_hist(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 2 HS histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])

    hist = np.concatenate((hist_h, hist_s))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_hs_concat_hist_blur(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 2 HS histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])

    hist = np.concatenate((hist_h, hist_s))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_hsv_concat_hist_blur(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 HSV histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [256], [0, 256])

    hist = np.concatenate((hist_h, hist_s, hist_v))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_tile_partition(img, r, c):
    (h, w) = img.shape[:2]
    padd_w = w // c
    padd_h = h // r
    return [img[padd_h * i:padd_h * (i + 1), padd_w * j:padd_w * (j + 1)]
            for i in range(r) for j in range(c)]


def get_hs_multi_hist(tiles, mask=None):
    ''' Paramenters: img (color image)
        Returns: return several hist concat result of splitting the image '''

    hist = []
    if mask is None:
        for tile in tiles:
            hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            hist.append(cv2.calcHist([hsv_tile], [0], mask, [256], [0, 256]))
            hist.append(cv2.calcHist([hsv_tile], [1], mask, [256], [0, 256]))
        hist = np.concatenate(hist)
    else:
        for i in range(len(tiles)):
            hsv_tile = cv2.cvtColor(tiles[i], cv2.COLOR_BGR2HSV)
            hist.append(cv2.calcHist([hsv_tile], [0], mask[i], [256], [0, 256]))
            hist.append(cv2.calcHist([hsv_tile], [1], mask[i], [256], [0, 256]))
        hist = np.concatenate(hist)

    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_hs_multiresolution_hist(img, mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # #get hist of the whole img
    hist = get_hs_concat_hist(img, mask)
    mask_tiled = None

    # get hist of the 2x2 partition
    tiles = get_tile_partition(img, 2, 2)
    if mask is not None:
        mask_tiled = get_tile_partition(mask, 2, 2)
    hist = np.concatenate((hist, get_hs_multi_hist(tiles, mask_tiled)))

    # get hist of the 4x4 partition
    tiles = get_tile_partition(img, 4, 4)
    if mask is not None:
        mask_tiled = get_tile_partition(mask, 4, 4)
    hist = np.concatenate((hist, get_hs_multi_hist(tiles, mask_tiled)))

    return hist


def get_multiresolution_hist(img, mask=None):
    # get hist of the whole img

    hist = get_bgr_concat_hist(img, mask)

    # get hist of the 2x2 partition
    tiles = get_tile_partition(img, 2, 2)
    mask_tiled = None

    if mask is None:
        tiles_hist = [get_bgr_concat_hist(tiles[i], None) for i in range(len(tiles))]
    else:
        mask_tiled = get_tile_partition(mask.copy(), 2, 2)
        tiles_hist = [get_bgr_concat_hist(tiles[i], mask_tiled[i]) for i in range(len(tiles))]
    hist = np.concatenate((hist, *tiles_hist))

    # get hist of the 4x4 partition
    tiles = get_tile_partition(img, 4, 4)
    if mask is None:
        tiles_hist = [get_bgr_concat_hist(tiles[i], None) for i in range(len(tiles))]
    else:
        mask_tiled = get_tile_partition(mask, 4, 4)
        tiles_hist = [get_bgr_concat_hist(tiles[i], mask_tiled[i]) for i in range(len(tiles))]
    hist = np.concatenate((hist, *tiles_hist))

    return hist


def get_gray_multiresolution_hist(img, mask=None):
    # get hist of the whole img

    hist = get_gray_hist(img, mask)

    # get hist of the 2x2 partition
    tiles = get_tile_partition(img, 2, 2)
    mask_tiled = None

    if mask is None:
        tiles_hist = [get_gray_hist(tiles[i], None) for i in range(len(tiles))]
    else:
        mask_tiled = get_tile_partition(mask.copy(), 2, 2)
        tiles_hist = [get_gray_hist(tiles[i], mask_tiled[i]) for i in range(len(tiles))]
    hist = np.concatenate((hist, *tiles_hist))

    # get hist of the 4x4 partition
    tiles = get_tile_partition(img, 4, 4)
    if mask is None:
        tiles_hist = [get_gray_hist(tiles[i], None) for i in range(len(tiles))]
    else:
        mask_tiled = get_tile_partition(mask, 4, 4)
        tiles_hist = [get_gray_hist(tiles[i], mask_tiled[i]) for i in range(len(tiles))]
    hist = np.concatenate((hist, *tiles_hist))

    return hist


def get_hs_concat_hist_st(img, mask=None):
    ''' Paramenters: img (color image)
        Returns: numpyarray with the 3 HSV histograms concatenated '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
    hist_concat = np.concatenate((hist_h, hist_s))
    hsv_st = linear_stretch(hsv, hist_concat)
    hist_h = cv2.calcHist([hsv_st], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([hsv_st], [1], mask, [256], [0, 256])

    hist = np.concatenate((hist_h, hist_s))
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2, alpha=1.)
    return hist


def get_descriptors(img, mask=None):
    """ Paramenters: img (color image)
        Returns: descript_dic (dictionary with descriptors names as keys) """
    descript_dic = {}
    # descript_dic['gray_hist'] = get_gray_hist(img, mask)
    # descript_dic['bgr_concat_hist'] = get_bgr_concat_hist(img, mask)
    # descript_dic['cielab_concat_hist'] = get_bgr_concat_hist(img, mask)
    # descript_dic['ycrcb_concat_hist'] = get_ycrcb_concat_hist(img, mask)
    # descript_dic['sift'] = get_sift_desc(img, None)
    descript_dic['orb'] = get_orb_desc(img, mask)
    # descript_dic['daisy'] = get_daisy_desc(img)
    # descript_dic['lbp'] = get_lbp(img)
    # descript_dic['hsv_concat_hist'] = get_hsv_concat_hist(img, mask)
    # descript_dic['hs_concat_hist'] = get_hs_concat_hist(img, mask)
    # descript_dic['DCT-16'] = get_DCT_coefs(img, N=16)
    # descript_dic['DCT-32'] = get_DCT_coefs(img, N=32)
    # descript_dic['DCT-16-8'] = get_DCT_coefs(img, N=16)
    # descript_dic['DCT-16-32'] = get_DCT_coefs(img, N=16, block_w=32)
    # descript_dic['DCT-16-64'] = get_DCT_coefs(img, N=16, block_w=64)
    # descript_dic['hs_concat_hist_st'] = get_hs_concat_hist_st(img, mask)
    # descript_dic['hs_concat_hist_blur'] = get_hs_concat_hist_blur(img, mask)
    # descript_dic['hsv_concat_hist_blur'] = get_hsv_concat_hist_blur(img, mask)
    # descript_dic['h_multi_hist'] = get_h_multi_hist(img, mask)
    tiles = get_tile_partition(img, 2, 2)
    mask_tiled = None
    if mask is not None:
        mask_tiled = get_tile_partition(mask, 2, 2)

    # descript_dic['hs_multi_hist'] = get_hs_multi_hist(tiles, mask_tiled)
    # descript_dic['hs_multiresolution'] = get_hs_multiresolution_hist(img, mask)
    # descript_dic['bgr_multiresolution'] = get_multiresolution_hist(img, mask)
    descript_dic['hsv_multiresolution'] = get_multiresolution_hist(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), mask)
    descript_dic['hog'] = get_hog(img)
    # lbp_im = get_lbp(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # descript_dic['lbp_multiresolution'] = get_gray_multiresolution_hist(lbp_im, mask)
    # descript_dic['lbp_hist'] = get_gray_hist(lbp_im, mask)
    # descript_dic['dct-200'] = get_dct(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100)
    # descript_dic['dct-150'] = get_dct(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 150)
    descript_dic['author'] = descript_dic['title'] = ''

    return descript_dic

# import distance_metrics as dist 

# cv2.imshow('im', lbim)
# cv2.waitKey(0)

# h = get_gray_multiresolution_hist(lbim)
# # h = get_bgr_concat_hist(lbim)
# print('hi')
# print(h.shape)

# # dist.display_comparison(h,h)
