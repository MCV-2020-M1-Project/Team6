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
import random as rnd
import numpy as np
import cv2
import pickle as pkl


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


def get_all_measures(a, b, display=False):
    '''
    Return a dictionary with all available measures. Keys are:
    * 'eucl': Euclidean distance
    * 'l1': L1 distance
    * 'x2': X2 distance
    * 'h_inter': Histogram intersection (similarity)
    * 'hell_ker': Hellinger kernel (similarity)
    * 'corr': correlation
    '''
    measures = {
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
