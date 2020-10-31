'''
Functions calculating different possible difference/similarity metrics

Metrics implemented:

* Euclidean distance
* L1 distance
* X2 distance
* Histogram intersection (similarity)
* Hellinger kernel (similarity)

201007 - For now assuming descriptor is a 1d numpy float array
'''
from difflib import SequenceMatcher
import numpy as np
import cv2

def gestalt(a, b):
    '''
    Returns difference ration between two strings
    '''
    if a is None:
        a = ''
    if b is None:
        b = ''

    result = 1 - SequenceMatcher(None, a, b).ratio()
    # if result < 0.5:
    #     print(a, 'vs', b)
    #     print(result)
    return result


def longest_common_subsequence(X, Y, m, n):
    '''
    From: https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
    '''
    if m == 0 or n == 0:
        return 0
    elif X[m-1] == Y[n-1]:
        return 1 + longest_common_subsequence(X, Y, m-1, n-1)
    else:
        return max(longest_common_subsequence(X, Y, m, n-1), longest_common_subsequence(X, Y, m-1, n))


def longest_common_substring(X, Y, m, n):
    '''
    From: https://www.geeksforgeeks.org/longest-common-substring-dp-29/
    '''
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]

    result = 0

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i-1] == Y[j-1]:
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result


def levenshtein(str1, str2):
    '''
    From: https://es.wikipedia.org/wiki/Distancia_de_Levenshtein#El_algoritmo 
    '''
    d=dict()
    for i in range(len(str1)+1):
        d[i]=dict()
        d[i][0]=i
    for i in range(len(str2)+1):
        d[0][i] = i
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            d[i][j] = min(d[i][j-1]+1, d[i-1][j]+1, d[i-1][j-1]+(not str1[i-1] == str2[j-1]))
    return d[len(str1)][len(str2)]


def hamming(string_a, string_b):
    '''
    From: my brain (that's why its so lame)
    '''
    if len(string_a) != len(string_b):
        print('Error: String lenghts should match')
        return -1
    result = 0
    for a, b in zip(string_a, string_b):
        if a.lower() != b.lower():
            result += 1
    return result / len(string_a)


def compare_text(a, b):
    '''
    Just trying out a bunch of measures, the lower they are the better
    (still have to thing how to use the longsest comon stuff)

    The best one is porbably gestalt (the one python uses in sequence matcher)
    '''
    print(a, b, len(a), len(b))
    results = []
    results.append(hamming(a, b))
    results.append(levenshtein(a, b))
    results.append( max(len(a), len(b))-longest_common_substring(a,b, len(a), len(b)))
    results.append( max(len(a), len(b))-longest_common_subsequence(a,b, len(a), len(b)))
    results.append(round(gestalt(a, b), 2))

    return results


def get_euclidean_distance(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns Euclidean distance (pylint get off my back)
    '''
    dif = descriptor_a - descriptor_b
    return np.sqrt(np.sum(dif*dif))


def get_l1_distance(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns L1 distance
    '''
    return np.sum(abs(descriptor_a - descriptor_b))


def get_x2_distance(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns X2 distance
    '''
    dif = descriptor_a - descriptor_b

    num = dif*dif
    den = descriptor_a+descriptor_b
    return np.sum(np.divide(num, den, out=np.zeros_like(num), where=den != 0))


def get_hist_intersection(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns histogram intersection
    '''
    return np.sum(np.minimum(descriptor_a, descriptor_b))


def get_hellinger_kernel(descriptor_a, descriptor_b):
    '''
    Gets descriptors as numpy arrays and returns hellinger kernel (whatever that is)
    '''
    if any(descriptor_a < 0) or any(descriptor_b < 0):
        print('All descriptor entries should be positive')
        return -1
    return np.sum(np.sqrt(descriptor_a*descriptor_b))


def get_correlation(a, b):  
    '''
    Correlation, implemented according to opencv documentation on histogram comparison
    '''
    dev_a = (a - np.mean(a))
    dev_b = (b - np.mean(b))

    return np.sum(dev_a*dev_b) / np.sqrt(np.sum(dev_a*dev_a)*np.sum(dev_b*dev_b))


def display_comparison(a, b):
    '''
    Displays an iamge with both descriptors (as histograms) alongside calculated measures

    This is kind of a silly thing, more showy than anything, but it might be useful when triying to
    decide which distance works best
    '''
    # image
    display_m_img = np.zeros((460, 828, 3), dtype=np.uint8)

    distances = get_all_measures(a, b)
    # measures
    text = [
            #'Euclidean: ' + str(round(distances['eucl'], 2)),
            'X2: ' + str(round(distances['x2'], 2)),
            'L1: ' + str(round(distances['l1'], 2)) 
            # 'Hist intersection: ' + str(round(distances['h_inter'], 2)),
            # 'Hellinger Kernel: ' + str(round(distances['hell_ker'], 2)),
            # 'Correlation: ' + str(round(distances['corr'], 2)),
            # 'Chi square:' + str(round(distances['chisq'], 2))
        ]

    # Draw histograms
    ## Some position parameters
    hist_sq_size = (512, 200)

    x_offset = 20
    bt_y_hist_1 = 220
    bt_y_hist_2 = 440

    measure_text_pos = (552, 20)

    ## Draw first hist
    for k, v in enumerate(a):
        cv2.line(display_m_img, (int(hist_sq_size[0]*k/len(a)) + x_offset, bt_y_hist_1),
                                (int(hist_sq_size[0]*k/len(a)) + x_offset, bt_y_hist_1 - int(hist_sq_size[1]*v/max(a))),
                                (0, 255, 0)
                                )

    ## Draw second hist
    for k, v in enumerate(b):
        cv2.line(display_m_img, (int(hist_sq_size[0]*k/len(b)) + x_offset, bt_y_hist_2), 
                                (int(hist_sq_size[0]*k/len(b)) + x_offset, bt_y_hist_2 - int(hist_sq_size[1]*v/max(b))),
                                (0, 0, 255)
                                )

    ## Display text
    y = measure_text_pos[1]
    for t in text:
        cv2.putText(display_m_img, t, (measure_text_pos[0], y), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 0, 0))
        y += 15

    cv2.imshow('Display', display_m_img)
    cv2.waitKey(0)

    return


def get_all_measures(a, b, display=False, text=False):
    '''
    Return a dictionary with all available measures. Keys are:
    * 'eucl': Euclidean distance
    * 'l1': L1 distance
    * 'x2': X2 distance
    * 'h_inter': Histogram intersection (similarity)
    * 'hell_ker': Hellinger kernel (similarity)
    * 'corr': correlation
    '''
    if text:
        measures =  {
                    'gestalt': gestalt(a, b)
                    # 'levenshtein':levenshtein(a,b),
                    # 'hamming': hamming(a, b)
                    }
        # print(measures['gestalt'])
    else:
        measures =  {
                    # 'eucl': get_euclidean_distance(a, b),
                    'l1': get_l1_distance(a, b),
                    'x2': get_x2_distance(a, b),
                    # 'h_inter': get_hist_intersection(a, b),
                    # 'hell_ker': get_hellinger_kernel(a, b), 
                    # 'corr': get_correlation(a, b),
                    # 'chisq': get_chisq_distance(a, b)
                    }

    if display:
        for k, v in measures.items():
            print(k + ':', '{:.2f}'.format(v))

    return measures
