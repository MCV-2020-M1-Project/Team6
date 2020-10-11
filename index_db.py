import pickle as pkl
import cv2
import descriptor_lib as desc
import os

#Generate descriptors for all images in the BBDD
descript_dic_list = []

for i in range(287):
    img_path = ['..', 'datasets', 'BBDD', 'bbdd_' + '{:05d}'.format(i) +'.jpg']
    img = cv2.imread(os.path.join(*img_path), cv2.IMREAD_COLOR)
    descript_dic_list.append(desc.get_descriptors(img)) #get a dic with the descriptors for the img

#Save descriptors
save_path = ['pkl_data', 'bd_descriptors.pkl']

if not os.path.isdir(save_path[0]):
    os.mkdir(save_path[0])

with open(os.path.join(*save_path), 'wb') as dbfile:
    pkl.dump(descript_dic_list, dbfile)
