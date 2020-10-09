import pickle
import cv2
import descriptor_lib as desc
import sys


def main(img_name):

    #Read descriptors of the museum db from .pkl
    db_descript_dic = {}
    with open('bd_descriptors.pkl', 'rb') as dbfile:
        pickle.dump(db_descript_dic, dbfile)
    
    #Get a dic with the descriptors of the query image
    query_img = cv2.imread('../qsd1_w1/'+ img_name +'.jpg', cv2.IMREAD_GRAYSCALE)
    query_descript_dic = desc.get_descriptors(query_img)

    ##task 2##
    ##task 3##

    return



if __name__ == "__main__":

    if len(sys.argv) >= 2:
        img_name = sys.argv[1]
        main(img_name)
    else:
        print("A query img is required.")