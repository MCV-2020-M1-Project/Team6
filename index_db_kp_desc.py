import os
import pickle as pkl
import cv2
from libs import kp_descriptors

#Generate descriptors for all images in the BBDD
descript_dic_list = []

for i in range(287):
    if i % 20 == 0: print(i)
    img_path = ['..', 'datasets', 'BBDD', 'bbdd_' + '{:05d}'.format(i) + '.jpg']
    img = cv2.imread(os.path.join(*img_path), cv2.IMREAD_COLOR)
    temp_dict={}
    temp_dict['sift'] = kp_descriptors.get_SIFT_desc(img)
    temp_dict['orb'] = kp_descriptors.get_ORB_desc(img)
    # temp_dict['surf'] = kp_descriptors.get_SURF_desc(img)
    print(os.path.join(*img_path))
    # print(temp_dict.values())
    temp_dict['idx'] = i
    descript_dic_list.append(temp_dict)  # get a dic with the descriptors for the img

# Save descriptors
save_path = ['pkl_data', 'kp_bd_descriptors.pkl']

if not os.path.isdir(save_path[0]):
    os.mkdir(save_path[0])

with open(os.path.join(*save_path), 'wb') as dbfile:
    pkl.dump(descript_dic_list, dbfile)