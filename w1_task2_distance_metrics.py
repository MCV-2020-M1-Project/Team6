'''
Functions calculating different possible difference/similarity metrics

Metrics implemented:

* Euclidean distance
* L1 distance
* X² distance
* Histogram intersection (similarity)
* Hellinger kernel (similarity)

201007 - For now assuming descriptor is a 1d numpy float array
'''
import random as rnd
import numpy as np


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
    return np.sum(dif*dif/(descriptor_a+descriptor_b))


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

# at least it runs?
a = np.array([rnd.uniform(0, 100) for i in range(255)])
b = np.array([rnd.uniform(0, 100) for i in range(255)])

print('Euclidean:', get_euclidean_distance(a, a))
print('L1:', get_l1_distance(a, b))
print('X2:', get_x2_distance(a, b))
print('Histogram intersection:', get_hist_intersection(a, b))
print('Hellinger kernel:', get_hellinger_kernel(a, b))
