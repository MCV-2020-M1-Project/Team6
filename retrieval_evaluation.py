import ml_metrics as metrics
import pickle as pkl
import cv2
import argparse
import cbir
import descriptor_lib as desc
import os, os.path
import csv


#calculates mean of all mapk values for particular method
def calculate_mean_all(mapk_values):
    mean_of_mapk=sum(mapk_values)/len(mapk_values)
    return mean_of_mapk

def main(queryset_name, descriptor, measure, k, similarity):

    #Read descriptors of the museum db from .pkl
    path = ['pkl_data','bd_descriptors.pkl'] #for making the path system independent
    db_descript_list = []
    with open(os.path.join(*path), 'rb') as dbfile:
        db_descript_list = pkl.load(dbfile)

    # reading images from queryset
    path = ['..','datasets', queryset_name]
    qs_number = len([name for name in os.listdir(os.path.join(*path)) \
        if not '.pkl' in name])

    #Get a dic with the descriptors of the images in the query set
    qs_descript_list = []
    for i in range(qs_number):
        path = ['..','datasets', queryset_name, '{:05d}'.format(i)+'.jpg']
        img = cv2.imread(os.path.join(*path), cv2.IMREAD_COLOR)
        if img is None:
            print('Error reading image', os.path.join(*path))
            quit()
        qs_descript_list.append(desc.get_descriptors(img)) #get a dic with the descriptors for the img

    predicted = [] #order predicted list of images for the method used on particular image
    #Get the results for every image in the query dataset
    for query_descript_dic in qs_descript_list:
        predicted.append(cbir.get_histogram_top_k_similar( \
            query_descript_dic[descriptor], db_descript_list, descriptor, measure, similarity, k))

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
    parser.add_argument('-d', required=False, default='bgr_concat_hist', type=str)
    parser.add_argument('-m', required=False, default='corr', type=str)
    parser.add_argument('-k', required=False, default='5', type=int)
    parser.add_argument('-s', '--similarity', required=False, default=False, action='store_true')

    args = parser.parse_args()

    main(args.q, args.d, args.m, args.k, args.similarity)