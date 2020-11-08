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
import numpy as np
# Our code imports
import cbir
from libs import descriptors as desc
from libs import background_removal as bg
from libs import box_retrieval as boxret
from libs import denoising as dn
from libs import text_retrieval as txt


def sort_rects_lrtb(rect_list):
    '''
    Sorts rect lists from left to right and top to bottom
    '''
    return sorted(rect_list, key = lambda x: (x[0], x[1]))


def main(queryset_name, descriptor, measure, k, similarity, background, bbox, ocr, desc_check):
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

        # # end ignore -1
        # if i in {0, 2, 3, 4, 9, 12, 14, 18, 20, 21, 27}: 
        #     continue
        # # end ignore -1

        path = ['..','datasets', queryset_name, '{:05d}'.format(i)+'.jpg']
        img = cv2.imread(os.path.join(*path), cv2.IMREAD_COLOR)
        if img is None:
            print('Error reading image', os.path.join(*path))
            quit()

        paintings = []

        if background:
            # masks = bg.method_canny(img)
            # _, masks = bg.hsv_thresh_method(img.copy(), 2)
            masks = bg.method_canny_multiple_paintings(img.copy())[1]

            masks = sort_rects_lrtb(masks)
            for mask in masks:
                if abs(mask[1] - mask[3]) < 50: # wrong crop
                    continue
                v1 = mask[:2]
                v2 = mask[2:]
                paintings.append(img[v1[1]:v2[1], v1[0]:v2[0]])
        else:
            paintings = [img]
        
        # print(i, len(paintings))

        # for i, p in enumerate(paintings):
        #     cv2.imshow('crop' + str(i), p)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Keypoint stuff

        if bbox:
            box_masks = []

            for painting in paintings:
                grad_method = boxret.get_boxes(painting.copy())
                if grad_method is None:
                    print('grad method failed')
                    sat_method = boxret.filled_boxes(painting.copy())[1]
                    box_masks.append(1- sat_method)
                else:
                    box_masks.append(1- grad_method)
            # box_masks = [(1 - boxret.get_boxes(painting.copy())) if boxret.get_boxes(painting.copy()) is not None else (1 - boxret.filled_boxes(painting.copy())[1]) \
                # for painting in paintings]

            text_list = []
            txt_file = open('ocr_results/{:05d}.txt'.format(i), 'w')
            # OCR
            for bb, p in zip(box_masks, paintings):
                mask = 1 - bb

                # cv2.imshow('ocr', cv2.resize(p, (500, 500*p.shape[0]//p.shape[1])))
                # cv2.imshow('Mask', cv2.resize(255*mask, (500, 500*p.shape[0]//p.shape[1])))

                a = np.where(mask > 0)
                pts = [(i, j) for i,j in zip(*a)]

                if len(pts) == 0 or not ocr:
                    text_list.append('')
                    txt_file.write('\n')
                    continue
                # print(pts[0], pts[-1])

                masked_img = p[pts[0][0]:pts[-1][0], pts[0][1]:pts[-1][1]]

                # masked_img = cv2.bitwise_and(p, p, mask=mask)
                # masked_img = p[bb > 0]
                text = txt.get_text(masked_img.copy())
                # print(text)
                # cv2.waitKey(0)
                txt_file.write(text)
                txt_file.write('\n')
                text_list.append(text)
            # author = ocr(im, box_masks[i])
            txt_file.close()
            # Denoise image
            paintings = [dn.denoise_img(painting) for painting in paintings]

            # get a dict with the descriptors for the n pictures per painting
            temp_list = []
            for i in range(len(paintings)):
                d = desc.get_descriptors(paintings[i].copy(), None) # box_masks[i]) #TODO: Trying not giving masks, as boxes are transparent
                d['author'] = d['title'] = text_list[i]
                temp_list.append(d)
                # print(d['author'])
                # cv2.waitKey(0)
            qs_descript_list.append(temp_list)

            # qs_descript_list.append([desc.get_descriptors(paintings[i].copy(), box_masks[i]) \
            #  for i in range(len(paintings))])

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
        predicted.append([cbir.get_top_k_multi(p, \
                        db_descript_list,  ['hog', 'hsv_multiresolution'], [0.5, 0.5], \
                        measure, similarity, k, {'author': 0.3}, desc_check=desc_check) \
                        for p in query_descript_dic])
    # ['hog', 'hsv_multiresolution', 'DCT-16-64'], [0.5, 0.5, 0],
    # print(predicted)

    # For generating submission pkl
    with open('../dlcv06/m1-results/week4/QST1/method1/result.pkl', 'wb') as f:
        print('Pickles...')
        pkl.dump(predicted, f)
        print('...gonna pick')
    quit()

    #Read groundtruth from .pkl
    actual = [] #just a list of all images from the query folder - not ordered
    path = ['..','datasets', queryset_name, 'gt_corresps.pkl']
    with open(os.path.join(*path), 'rb') as gtfile:
        actual = pkl.load(gtfile)

    # # ignore -1
    # new_actual = []
    # for c in actual:
    #     if c[0] != -1:
    #         new_actual.append(c)
    # actual = new_actual
    # # end ignore -1

    # Extending lists to get performance for list of lists of lists
    new_predicted = []
    for images, actual_im in zip(predicted, actual):
        if len(images) > len(actual_im):
            images = images[:len(actual_im)]
        elif len(images) < len(actual_im):
            for i in range(len(actual_im) - len(images)):
                images.append([0])
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

    print('actual:', actual)
    # print('new actual:', new_actual)
    print('predicted:', predicted)
    # print('new predicted:', new_predicted)
    print('Result =', map_k)

    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Queryset: ' + queryset_name, 'Descriptor: ' + descriptor, \
            'Measure: ' + measure, 'k: '+ str(k)])
        writer.writerow(['Map@k: '+str(map_k)])
        writer.writerow(['Actual','Predicted'])
        for i in range(len(actual)):
            writer.writerow([str(actual[i]), str(predicted[i])])

    print('='*20)
    print('Actual, predcted')
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for a, p in zip(new_actual, new_predicted):
        a = a[0]
        p = p[0]
        # print(a, p)
        if p == -1: # positive
            if p == a:
                tp += 1
            else:
                fp += 1
        else: #negative
            if a == -1:
                fn += 1
            else:
                tn += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall/(precision + recall)

    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    print(f'Precision: {precision}\nRecall: {recall}\n F1 {f1}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', required=True, type=str, help='query set')
    parser.add_argument('-d', required=False, default='hs_multi_hist', type=str)
    parser.add_argument('-m', required=False, default='l1', type=str)
    parser.add_argument('-k', required=False, default=5, type=int)
    parser.add_argument('-b', '--background', required=False, default=False, action='store_true')
    parser.add_argument('-s', '--similarity', required=False, default=False, action='store_true')
    parser.add_argument('-bb', '--bbox', required=False, default=False, action='store_true')
    parser.add_argument('-o', '--ocr', required=False, default=False, action='store_true')
    parser.add_argument('-dc', '--desc-check', required=False, default=False, action='store_true')

    args = parser.parse_args()

    main(args.q, args.d, args.m, args.k, args.similarity, args.background, args.bbox, args.ocr, args.desc_check)
