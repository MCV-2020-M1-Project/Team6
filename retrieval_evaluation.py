import ml_metrics as metrics
import pickle as pkl
import cv2
import argparse
import cbir
import descriptor_lib as desc
import background_removal as bg 
import os, os.path
import csv
import numpy as np
import box_retrieval

#calculates mean of all mapk values for particular method
def calculate_mean_all(mapk_values):
    mean_of_mapk=sum(mapk_values)/len(mapk_values)
    return mean_of_mapk

def main(queryset_name, descriptor, measure, k, similarity, background, bbox):

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
            # placeholder call to bg removal 
            # mask = bg.method_similar_channels_jc(img, 30)
            # masks = bg.method_canny(img)
            masks = bg.hsv_thresh_method(img)[1]
            # masks = bg.method_canny_multiple_paintings(img)[1]
            # print(len(masks), masks)
            for mask in masks:
                v1 = mask[:2] # remember it was [1][2:] before
                v2 = mask[2:]
                paintings.append(img[v1[1]:v2[1], v1[0]:v2[0]])
        else:
            paintings = [img]

        if bbox:
            # box_masks = [(1 - box_retrieval.filled_boxes(painting.copy())[1]) for painting in paintings]
            # qs_descript_list.append([desc.get_descriptors(paintings[i], box_masks[i]) \
            #  for i in range(len(paintings))]) # get a dic with the descriptors for the n pictures per painting

            box_masks = [(box_retrieval.filled_boxes(painting.copy())[1]) for painting in paintings]
            for i in range(len(box_masks)):
                if np.mean(box_masks[i]) == 0:
                    box_masks[i] = (box_retrieval.get_boxes(paintings[i]))
            print(np.uint8(box_masks[i]))
            box_masks[i] = np.uint8(box_masks[i])
            cv2.imshow('my',box_masks[i]*255)
            cv2.waitKey(0)
            inpainted_paintings = []
            for i in range(len(paintings)):
                box_masks[i] = cv2.resize(box_masks[i], (512,512*box_masks[i].shape[0]//box_masks[i].shape[1]))
                box_masks[i] = cv2.morphologyEx(box_masks[i],cv2.MORPH_DILATE, np.ones((7,7)), iterations=3)
                paintings[i] = cv2.resize(paintings[i], (512,512*paintings[i].shape[0]//paintings[i].shape[1]))
                paintings[i] = cv2.inpaint(paintings[i],box_masks[i],3,cv2.INPAINT_NS)
                inpainted_paintings.append(paintings[i])
            cv2.imshow('inapinted',inpainted_paintings[0])
            cv2.waitKey(0)
            #quit()
            qs_descript_list.append([desc.get_descriptors(painting) for painting in inpainted_paintings])
            
            temp_list = []
            for i in range(len(paintings)):
                bbox_loc =  box_retrieval.filled_boxes(paintings[i].copy())[4]
                mask_loc = masks[i] if background else (0, 0)

                bbox_loc[0] += mask_loc[0]
                bbox_loc[1] += mask_loc[1]
                bbox_loc[2] += mask_loc[0]
                bbox_loc[3] += mask_loc[1]
                
                if i > 0: # maybe check this in the future
                    if temp_list[0][0] + temp_list[0][1] < bbox_loc[0]+bbox_loc[1]:
                        temp_list.append(bbox_loc)
                    else:
                        temp_list.insert(0, bbox_loc)
                else:
                    temp_list.append(bbox_loc)
            
            bbox_list.append(temp_list)
        else:
            qs_descript_list.append([desc.get_descriptors(paintings[i], None) \
             for i in range(len(paintings))]) # get a dic with the descriptors for the n pictures per painting
    
    with open('text_boxes.pkl', 'wb') as f:
        pkl.dump(bbox_list, f)
    
    predicted = [] #order predicted list of images for the method used on particular image
    #Get the results for every image in the query dataset
    # for query_descript_dic in qs_descript_list:
    #     predicted.append(cbir.get_histogram_top_k_similar( \
    #         query_descript_dic[descriptor], db_descript_list, descriptor, measure, similarity, k))

    # n images per painting
    for query_descript_dic in qs_descript_list:
        predicted.append([cbir.get_histogram_top_k_similar(p[descriptor], \
                        db_descript_list, descriptor, measure, similarity, k) \
                        for p in query_descript_dic][0]) # IF GT FORMAT IS AS IN W1, REMEMBER TO INDEX THE FIRST (AND ONLY) ELEMENT OF THIS COMPRESSED LIST

    #Read grandtruth from .pkl
    actual = [] #just a list of all images from the query folder - not ordered
    path = ['..','datasets', queryset_name, 'gt_corresps.pkl']
    with open(os.path.join(*path), 'rb') as gtfile:
        actual = pkl.load(gtfile)

    map_k = metrics.kdd_mapk(actual,predicted,k)

    print('actual:', actual)
    print('predicted:', predicted)
    print(map_k)

    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Queryset: '+queryset_name, 'Descriptor: '+descriptor, 'Measure: '+ measure, 'k: '+ str(k)])
        writer.writerow(['Map@k: '+str(map_k)])
        writer.writerow(['Actual','Predicted'])
        for i in range(len(actual)):
            writer.writerow([str(actual[i]), str(predicted[i])])

    #calculate_mean_all(mapk_values)


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