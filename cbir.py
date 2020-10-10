import pickle as pkl
import argparse
import cv2
import descriptor_lib as desc
import distance_metrics_lib as dists
import sys


def get_histogram_top_k_similar(query_descriptor, db_descriptor_list, k=3):
      
    distances_dict = {}
    idx = 0
    for db_point in db_descriptor_list:
        distances = dists.get_all_measures(query_descriptor, db_point[DESCRIPTOR])
        distances_dict[idx] = distances[MEASURE]
        idx += 1

    result = [key for key in sorted(distances_dict, key=distances_dict.get, reverse=True)[:k]]

    return result


def main(img_name):

    #Read descriptors of the museum db from .pkl
    db_descript_list = []
    with open('pkl_data/bd_descriptors.pkl', 'rb') as dbfile:
        db_descript_list = pkl.load(dbfile)

    #Get a dic with the descriptors of the query image
    query_img = cv2.imread('../datasets/qsd1_w1/'+ img_name + '.jpg', cv2.IMREAD_COLOR)
    query_descript_dic = desc.get_descriptors(query_img)

    ##task 3##
    result = get_histogram_top_k_similar(query_descript_dic[DESCRIPTOR], db_descript_list)

    print('result:', result)

    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='Image')
    parser.add_argument('-d', required=False, default='bgr_concat_hist', type=str)
    parser.add_argument('-m', required=False, default='corr', type=str)
    args = parser.parse_args()
    

    DESCRIPTOR = args.d
    MEASURE = args.m

    main(args.i)
