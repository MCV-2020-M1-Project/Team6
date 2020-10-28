'''
Main script for getting results and evaluation performance
'''
# Standart imports
import pickle as pkl
import os
import csv
import argparse
# Lib imports
import ml_metrics as metrics
import cv2
# Our code imports
import cbir
from libs import descriptors as desc
from libs import background_removal as bg
from libs import box_retrieval as boxret
from libs import denoising as dn


def sort_rects_lrtb(rect_list):
    '''
    Sorts rect lists from left to right and top to bottom
    '''
    return sorted(rect_list, key = lambda x: (x[0], x[1]))


def main(queryset_name, descriptor, measure, k, similarity, background, bbox):
    '''
    Main function
    '''
    #Read descriptors of the museum db from .pkl
    path = ['pkl_data','bd_descriptors.pkl'] #for making the path system independent
    db_descript_list = []
    with open(os.path.join(*path), 'rb') as dbfile:
        db_descript_list = pkl.load(dbfile)

    # reading images from queryset
    path = ['..','datasets', queryset_name]
    qs_number = len([name for name in os.listdir(os.path.join(*path)) \
        if '.jpg' in name])

    #Get a dic with the descriptors of the images in the query set
    qs_descript_list = []

    bbox_list = []

    for i in range(qs_number):
        path = ['..','datasets', queryset_name, '{:05d}'.format(i)+'.jpg']
        img = cv2.imread(os.path.join(*path), cv2.IMREAD_COLOR)
        if img is None:
            print('Error reading image', os.path.join(*path))
            quit()

        paintings = []
        if background:
            # masks = bg.method_canny(img)
            _, masks = bg.hsv_thresh_method(img, 2)
            # masks = bg.method_canny_multiple_paintings(img.copy())[1]

            masks = sort_rects_lrtb(masks)
            for mask in masks:
                v1 = mask[:2]
                v2 = mask[2:]
                paintings.append(img[v1[1]:v2[1], v1[0]:v2[0]])
        else:
            paintings = [img]

        # Denoise image
        denoised_imgs = [dn.denoise_img(painting) for painting in paintings]
        paintings = [p['color'] for p in denoised_imgs]
        ocr_images = [p['ocr'] for p in denoised_imgs]

        if bbox:
            box_masks = [(1 - boxret.filled_boxes(painting.copy())[1]) \
                for painting in paintings]

            # get a dict with the descriptors for the n pictures per painting
            qs_descript_list.append([desc.get_descriptors(paintings[i].copy(), box_masks[i]) \
             for i in range(len(paintings))])

            # Save text boxes in a pkl for evaluation
            temp_list = []
            for l, painting in enumerate(paintings):
                bbox_loc =  boxret.filled_boxes(painting.copy())[4]
                mask_loc = masks[l] if background else (0, 0)

                bbox_loc[0] += mask_loc[0]
                bbox_loc[1] += mask_loc[1]
                bbox_loc[2] += mask_loc[0]
                bbox_loc[3] += mask_loc[1]

                temp_list.append(bbox_loc)

            bbox_list.append(sort_rects_lrtb(temp_list))
        else:
            # get a dic with the descriptors for the n pictures per painting
            qs_descript_list.append([desc.get_descriptors(painting.copy(), None) \
             for painting in paintings])

    with open('text_boxes.pkl', 'wb') as f:
        pkl.dump(bbox_list, f)

    predicted = []
    for query_descript_dic in qs_descript_list:
        predicted.append([cbir.get_histogram_top_k_similar(p[descriptor], \
                        db_descript_list, descriptor, measure, similarity, k) \
                        for p in query_descript_dic])

    # For generating submission pkl
    # with open('../dlcv06/m1-results/week2/QST2/method2/result.pkl', 'wb') as f:
    #     pkl.dump(predicted, f)
    # quit()

    #Read groundtruth from .pkl
    actual = [] #just a list of all images from the query folder - not ordered
    path = ['..','datasets', queryset_name, 'gt_corresps.pkl']
    with open(os.path.join(*path), 'rb') as gtfile:
        actual = pkl.load(gtfile)

    # Extending lists to get performance for list of lists of lists
    new_predicted = []
    for images in predicted:
        for paintings in images:
            new_predicted.append(paintings)

    new_actual = []
    for images in actual:
        if len(images) > 1:
            for paintings in images:
                new_actual.append([paintings])
        else:
            new_actual.append(images)

    map_k = metrics.kdd_mapk(new_actual,new_predicted,k)

    # print('actual:', actual)
    print('new actual:', new_actual)
    # print('predicted:', predicted)
    print('new predicted:', new_predicted)
    print('Result =', map_k)

    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Queryset: ' + queryset_name, 'Descriptor: ' + descriptor, \
            'Measure: ' + measure, 'k: '+ str(k)])
        writer.writerow(['Map@k: '+str(map_k)])
        writer.writerow(['Actual','Predicted'])
        for i in range(len(actual)):
            writer.writerow([str(actual[i]), str(predicted[i])])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', required=True, type=str, help='query set')
    parser.add_argument('-d', required=False, default='hs_multi_hist', type=str)
    parser.add_argument('-m', required=False, default='x2', type=str)
    parser.add_argument('-k', required=False, default=5, type=int)
    parser.add_argument('-b', '--background', required=False, default=False, action='store_true')
    parser.add_argument('-s', '--similarity', required=False, default=False, action='store_true')
    parser.add_argument('-bb', '--bbox', required=False, default=False, action='store_true')

    args = parser.parse_args()

    main(args.q, args.d, args.m, args.k, args.similarity, args.background, args.bbox)
