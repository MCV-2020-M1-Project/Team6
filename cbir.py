import os
import argparse
import pickle as pkl

import cv2

from libs import descriptors as desc
from libs import distance_metrics as dists
# from libs import background_removal as bg
# from libs import box_retrieval as boxret


def get_histogram_top_k_similar(query_descriptor, db_descriptor_list, descriptor, measure, similarity, k=3):
 
    distances_dict = {}
    idx = 0
    for db_point in db_descriptor_list:
        distances = dists.get_all_measures(query_descriptor, db_point[descriptor])
        distances_dict[idx] = abs(distances[measure])
        idx += 1

    # for key in sorted(distances_dict, key=distances_dict.get, reverse=similarity)[:k]:
    #     print(key, distances_dict[key])

    result = [key for key in sorted(distances_dict, key=distances_dict.get, reverse=similarity)[:k]]


    return result


def main(img_name, descriptor, measure, k, background, similarity):

    #Read descriptors of the museum db from .pkl
    db_descript_list = []
    pkl_path = ['pkl_data', 'bd_descriptors.pkl']
    with open(os.path.join(*pkl_path), 'rb') as dbfile:
        db_descript_list = pkl.load(dbfile)

    #Get a dic with the descriptors of the query image
    qimg_path = ['..','datasets', 'qsd1_w1', img_name + '.jpg']
    query_img = cv2.imread(os.path.join(*qimg_path), cv2.IMREAD_COLOR)

    if background:
        print('Placeholder for background_removal(query_img) call')
        pass # Call background removal function

    query_descript_dic = desc.get_descriptors(query_img)

    ##task 3##
    result = get_histogram_top_k_similar(query_descript_dic[descriptor], db_descript_list, \
        descriptor, measure, similarity, k)

    print('result:', result)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, type=str, help='Image')
    parser.add_argument('-d', '--descriptor', required=False, default='bgr_concat_hist', type=str)
    parser.add_argument('-m', '--measure', required=False, default='corr', type=str)
    parser.add_argument('-b', '--background', required=False, default=False, action='store_true')
    parser.add_argument('-s', '--similarity', required=False, default=False, action='store_true')
    parser.add_argument('-k', required=False, default='1', type=int)
    args = parser.parse_args()

    main(args.image, args.descriptor, args.measure, args.k, args.background, args.similarity)
