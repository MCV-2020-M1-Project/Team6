import pickle
import descriptor_lib as desc
import cv2

#generate descriptors for all images in the BBDD
img_list = [cv2.imread('../BBDD/bbdd_00'+ '{:03d}'.format(i) +'.jpg', cv2.IMREAD_GRAYSCALE)\
    for i in range(287)]
descriptors_dic = desc.get_descriptors(img_list)

#save descriptors
with open('bd_descriptors.pkl', 'wb') as dbfile:
    pickle.dump(descriptors_dic, dbfile)
