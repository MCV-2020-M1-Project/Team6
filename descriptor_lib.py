import cv2

def get_hist(img_list):
    hist_list = [cv2.calcHist([img_list[i]],[0],None,[256],[0,256])\
        for i in range(len(img_list))]
    return hist_list

def get_descriptors(img_list):
    descript_dic = {}
    descript_dic['hist'] = get_hist(img_list)
    return descript_dic