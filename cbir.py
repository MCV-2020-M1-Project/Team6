import pickle as pkl
import argparse
import cv2
import descriptor_lib as desc
import distance_metrics_lib as dists
import sys
import os


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
    pkl_path = ['pkl_data', 'bd_descriptors.pkl']
    with open(os.path.join(*pkl_path), 'rb') as dbfile:
        db_descript_list = pkl.load(dbfile)

    #Get a dic with the descriptors of the query image
    qimg_path = ['..','datasets', 'qsd1_w1', img_name + '.jpg']
    query_img = cv2.imread(os.path.join(*qimg_path), cv2.IMREAD_COLOR)

    if BACKGROUND:
        print('Placehold for background_removal(query_img) call')
        pass # Call background removal function

    query_descript_dic = desc.get_descriptors(query_img)

    ##task 3##
    result = get_histogram_top_k_similar(query_descript_dic[DESCRIPTOR], db_descript_list)
    print('result:', result)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, type=str, help='Image')
    parser.add_argument('-d', '--descriptor', required=False, default='bgr_concat_hist', type=str)
    parser.add_argument('-m', '--measure', required=False, default='corr', type=str)
    parser.add_argument('-b', '--background', required=False, default=False, action='store_true')
    args = parser.parse_args()

    BACKGROUND = args.background
    DESCRIPTOR = args.descriptor
    MEASURE = args.measure

    main(args.image)
