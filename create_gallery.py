from libs import steganography
import os,shutil
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from libs import descriptors

def create_steg_files():
    # reading images from queryset
    path = ['..', 'datasets', 'BBDD']
    qs_number = len([name for name in os.listdir(os.path.join(*path)) \
                     if '.jpg' in name])

    # Get a dic with the descriptors of the images in the query set

    for i in range(qs_number):
        # # end ignore -1
        # if i in {0, 2, 3, 4, 9, 12, 14, 18, 20, 21, 27}:
        #     continue
        # # end ignore -1
        print(i)

        path = ['..', 'datasets', 'BBDD', 'bbdd_{:05d}'.format(i) + '.jpg']
        path = os.path.join(*path)
        img = cv2.imread(os.path.join(*path), cv2.IMREAD_COLOR)
        output_path = ['..', 'datasets' , 'steganography', '{:05d}'.format(i) +'_output'+ '.png']
        output_path = os.path.join(*output_path)

        # unmerged_image = steganography.unmerge(Image.open(os.path.join(*path)))
        # unmerged_image.save(os.path.join(*output_path))
        os.system(f'python libs/steganography.py unmerge  --img {path} --output {output_path}')

def create_rooms(room_list,room_nr):
    path = ['..', 'datasets', 'Gallery', '{:02d}'.format(room_nr)]
    path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)
    path_in = ['..', 'datasets', 'steganography']
    path_in = os.path.join(*path_in)

    file_list = sorted([name for name in os.listdir(path_in) \
                        if '.jpg' in name])
    print(file_list)

    # Get a dic with the descriptors of the images in the query set

    for nr in room_list:
        path_in = ['..', 'datasets', 'BBDD', 'bbdd_{:05d}'.format(nr)+'.jpg']
        # path_in_stg = ['..', 'datasets', 'steganography', '{:05d}_output'.format(nr) + '.jpg']
        path_in = os.path.join(*path_in)
        # path_in_stg = os.path.join(*path_in_stg)

        print(path_in,path)
        shutil.copy(path_in,path)
        # shutil.copy(path_in_stg, path)
        # quit()

def filter_images():
    all_hists=[]
    path = ['..', 'datasets', 'steganography']
    path = os.path.join(*path)
    files_list = sorted([name for name in os.listdir(path) \
                     if '.jpg' in name])
    print(len(files_list))
    # print(files_list[])
    # quit()
    all_hists=[]

    for id, img in enumerate(files_list):
        path_in = path+'/'+img
        image = cv2.imread(path_in,cv2.IMREAD_COLOR)
        hist = descriptors.get_bgr_concat_hist(image)
        all_hists.append(np.float32(hist))
        # if img=='00000_output.jpg' :break

    X = np.array([h for h in all_hists])
    Z = np.vstack((h for h in all_hists))
    Z = np.float32(Z)
    # print(all_hists[:10])
    print('kmeans start')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    #
    compactness, labels, centers = cv2.kmeans(X,10,None,criteria,10,flags)
    #
    # A = Z[labels.ravel() == 0]
    # B = Z[labels.ravel() == 1]
    # print(A,"done")
    # print("labels=",labels)
    #plot
    room0=[]
    room1=[]
    room2=[]
    room3 = []
    room4 = []
    room5 = []
    room6 = []
    room7 = []
    room8 = []
    room9 = []

    for id, label in enumerate(labels.ravel()):
        if label == 0 : room0.append(id)
        if label == 1 : room1.append(id)
        if label == 2: room2.append(id)
        if label == 3: room3.append(id)
        if label == 4: room4.append(id)
        if label == 5: room5.append(id)
        if label == 6: room6.append(id)
        if label == 7: room7.append(id)
        if label == 8: room8.append(id)
        if label == 9: room9.append(id)
    print('room0=', room0)
    print('room1=', room1)
    print('room2=', room2)
    print('room3=', room3)
    print('room4=', room4)
    print('room5=', room5)
    print('room6=', room6)
    print('room7=', room7)
    print('room8=', room8)
    print('room9=', room9)

    create_rooms(room0,0)
    create_rooms(room1, 1)
    create_rooms(room2, 2)
    create_rooms(room3, 3)
    create_rooms(room4, 4)
    create_rooms(room5, 5)
    create_rooms(room6, 6)
    create_rooms(room7, 7)
    create_rooms(room8, 8)
    create_rooms(room9, 9)


    return all_hists

filter_images()


# x = np.random.randint(25,100,25)
# y = np.random.randint(175,255,25)
# z = np.hstack((x,y))
# z = z.reshape((50,1))
# z = np.float32(z)
#
# hist = filter_images()
# print (z[1])
# print (type(z))
# print(type(z[1]))
# print (type(hist))
# print(type(hist[0]))
# # print (hist[0])
#
# X = np.random.randint(25,50,(25,2))
# Y = np.random.randint(60,85,(25,2))
# Z = np.vstack((X,Y))
# # convert to np.float32
# Z = np.float32(Z)
# print(type(Z))
# # print(Z)
