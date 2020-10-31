import os
import pickle as pkl
import cv2
from libs import descriptors as desc

#Generate descriptors for all images in the BBDD
descript_dic_list = []

text_db = {}
for i in range(287):
    text_path = ['..', 'datasets', 'BBDD', 'bbdd_' + '{:05d}'.format(i) +'.txt']

    with open(os.path.join(*text_path), 'r', encoding='ISO-8859-1') as fp:
        t = fp.read()
        text_info = t.strip()[1:-1].replace("'",'').split(',')
        text_info = tuple([t.strip() for t in text_info])

        text_db[i] =  text_info


for i in range(287):
    if i % 20==0: print(i)
    img_path = ['..', 'datasets', 'BBDD', 'bbdd_' + '{:05d}'.format(i) +'.jpg']
    img = cv2.imread(os.path.join(*img_path), cv2.IMREAD_COLOR)
    temp_dict = desc.get_descriptors(img)

    if len(text_db[i]) != 2:
        temp_dict['author'] = temp_dict['title'] = ''
    else:
        temp_dict['author'] = text_db[i][0]
        temp_dict['title'] = text_db[i][1]
    
    temp_dict['idx'] = i
    descript_dic_list.append(temp_dict) #get a dic with the descriptors for the img

# Save descriptors
save_path = ['pkl_data', 'bd_descriptors.pkl']

if not os.path.isdir(save_path[0]):
    os.mkdir(save_path[0])

with open(os.path.join(*save_path), 'wb') as dbfile:
    pkl.dump(descript_dic_list, dbfile)
