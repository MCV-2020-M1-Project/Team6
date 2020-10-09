import pickle
import cv2
import descriptor_lib as desc

#Generate descriptors for all images in the BBDD
descript_dic_list = []

for i in range(287):
    img = cv2.imread('../BBDD/bbdd_00'+ '{:03d}'.format(i) +'.jpg', cv2.IMREAD_COLOR)
    descript_dic_list.append(desc.get_descriptors(img)) #get a dic with the descriptors for the img

#Save descriptors
with open('bd_descriptors.pkl', 'wb') as dbfile:
    pickle.dump(descript_dic_list, dbfile)
