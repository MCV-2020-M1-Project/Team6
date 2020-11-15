import os
import sys
sys.path.append(os.path.split(__file__)[0])

import argparse, os
import cv2 as cv
import numpy as np
# from "../evaluation" import mask_evaluation
import glob
import pickle as pkl

def save_masks(removal_method, input_folder):
    output_path = f'../datasets/masks_extracted/{removal_method}'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files_img = glob.glob(f'../datasets/{input_folder}/*.jpg')
    # print(files_img)

    images = [(cv.imread(i), i.split('/')[-1].split('.')[0]) for i in files_img]

    if removal_method == "canny":
        for i in range(len(images)):
            cv.imwrite(os.path.join(output_path, f"{i:05d}.png"),
                       method_canny(images[i]))
    elif removal_method == "similar_channel":
        for i in range(len(images)):
            cv.imwrite(os.path.join(output_path, f"{i:05d}.png"),
                       method_similar_channels_jc(images[i], 30))    
    elif removal_method == 'hsv_thresh':
        for i in range(len(images)):
            cv.imwrite(os.path.join(output_path, images[i][1] + '.png'),
                       255*hsv_thresh_method(images[i][0], 2)[0])
    elif removal_method == 'multi_canny':
        for i in range(len(images)):
            cv.imwrite(os.path.join(output_path, images[i][1] + '.png'),
                    255*method_canny_multiple_paintings(images[i][0])[0])


    else:
        av_methods = 'canny', 'similar_channel', 'hsv_thresh'
        print('Unknown removal method (available methods', ', '.join(av_methods), ')')
        return

    print(f"[INFO] Masks successfully stored in '{output_path}'")


def get_measures(name, mask):
    img_annotation = cv.imread(f'../datasets/qsd2_w1/{name}.png', 0)
    p, r, f = mask_evaluation.mask_evaluation(img_annotation, mask)

    measure_dict = {'name': name,
                    'precision': p,
                    'recall': r,
                    'F1_measure': f}

    measure_list = [name, p, r, f]

    return measure_dict


def show_image(im):
    im = cv.imread(f'../datasets/qsd2_w1/{im}.jpg')

    x, y, z = cv.split(im)

    print(type(x))
    cv.imshow('x', x)
    cv.imshow('y', y)
    cv.imshow('z', z)
    print(x)


def decide_best_rect(contour_list, n=1):

    all_rects = []
    rects = []
    if n == 1:
        rect = 0, 0, 0, 0
        max_area = 0
        for cont in contour_list:
            x, y ,w ,h = cv.boundingRect(cont)
            if w*h > max_area:
                rect = x, y, x + w, y + h
                max_area = w*h
            all_rects.append((x, y, x + w, y + h))
        rects.append(rect)

    if n == 2:
        # Consider largest contour to be the first painting
        reversed_sorted_contours = sorted(contour_list, key=len, reverse=True)
        first_contour = reversed_sorted_contours[0]
        second_contour = []
        for contour in reversed_sorted_contours:
            if not contours_overlap(first_contour, contour):
                second_contour = contour
                break

        list_of_painting_coordinates = []

        # First painting
        Ax, Ay, Aw, Ah = cv.boundingRect(first_contour)
        rects.append([Ax, Ay, Ax+Aw, Ay+Ah])

        # Second painting (not always there is a second painting)
        if len(second_contour) > 0:
            Bx, By, Bw, Bh = cv.boundingRect(second_contour)
            if (Bw * Bh) > ((Aw * Ah) / 100):  # area of the second painting is
                rects.append([Bx, By, Bx+Bw, By+Bh])

    return rects, all_rects


def morph_threshold_mask(im, n):
    struct_el = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    im_morph = cv.morphologyEx(im, cv.MORPH_CLOSE, struct_el)

    struct_el = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
    im_morph = cv.morphologyEx(im_morph, cv.MORPH_CLOSE, struct_el)

    # struct_el = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # im_morph = cv.morphologyEx(im_morph, cv.MORPH_ERODE, struct_el)

    contours, _ = cv.findContours(im_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    rect_list, all_rects = decide_best_rect(contours, n)

    # im_morph_show = cv.cvtColor(im_morph, cv.COLOR_GRAY2BGR)
    # for r in all_rects:
    #     im_morph_show = cv.rectangle(im_morph_show, (r[0], r[1]), (r[2], r[3]), (255,0,0), 2)

    # cv.imshow('rects', cv.resize(im_morph_show, (900, 900*im_morph_show.shape[0]//im_morph_show.shape[1])))
    # cv.waitKey(0)

    mask_im = np.zeros_like(im)
    for rect in rect_list:
        mask_im[rect[1]:rect[3], rect[0]:rect[2]] = 1

    return mask_im, rect_list # mask retrieval functions should always return lists now


def hsv_thresh_method(im, n=1):
    return morph_threshold_mask(method_colorspace_threshold(im.copy(), [0, 255], [100, 255], [0, 200], 'hsv'), n)
 

def method_colorspace_threshold(image, x_range, y_range, z_range, colorspace, save=False, generate_measures=False):
    """
    x = [bottom,top]
    y = [bottom,top]
    z = [bottom,top]

    bottom - top has value from 0-255

    colorspace = 'bgr','rgb','hsv','ycrcb','cielab','

    generate measures - generates measures if set to True instead of a mask. If you want to generate measures
                        against ground truth as image provide name of the imagea without extension

    return: mask or measures( if generate_measures = True)
    """
    masks_measures = []
    if generate_measures:
        name = image
        img = cv.imread(f'../datasets/qsd2_w1/{name}.jpg')  # BGR, float
    else:
        img = image

    if colorspace == 'bgr': pass
    if colorspace == 'rgb': img = cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
    if colorspace == 'hsv': img = cv.cvtColor(img, cv.COLOR_BGR2HSV, img)
    if colorspace == 'hls': img = cv.cvtColor(img, cv.COLOR_BGR2HLS, img)
    if colorspace == 'ycrcb': img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb, img)
    if colorspace == 'cielab': img = cv.cvtColor(img, cv.COLOR_BGR2Lab, img)
    if colorspace == 'xyz': img = cv.cvtColor(img, cv.COLOR_BGR2XYZ, img)
    if colorspace == 'yuv': img = cv.cvtColor(img, cv.COLOR_BGR2YUV, img)

    # mask color
    lower = np.array([x_range[0], y_range[0], z_range[0]])
    upper = np.array([x_range[1], y_range[1], z_range[1]])
    mask_matrix = cv.inRange(img, lower, upper)
    
    if save:
        if not os.path.exists(f'../datasets/masks_extracted/mst_{colorspace}/'):
            os.makedirs(f'../datasets/masks_extracted/mst_{colorspace}/')
        cv.imwrite(f'../datasets/masks_extracted/mst_{colorspace}/{name}.png', mask_matrix)

    if generate_measures:
        return get_measures(name, mask_matrix)
    else:
        return np.uint8(mask_matrix / 255)


def contours_overlap(contour_A, contour_B):
    # Coordinates of bounding rectangle 1
    Ax, Ay, Aw, Ah = cv.boundingRect(contour_A)

    # Coordinates of bounding rectangle 2
    Bx, By, Bw, Bh = cv.boundingRect(contour_B)

    # If rectangle B is on the left of rectangle A
    if (Ax >= (Bx + Bw) or Bx >= (Ax + Aw)):
        return False

    # If rectangle B is above rectangle A
    if (Ay >= (By + Bh) or By >= (Ay + Ah)):
        return False

    return True


def method_canny_multiple_paintings_old(image, save=False, generate_measures=False):

    if generate_measures:
        name = image
        img = cv.imread(f'../datasets/qsd2_w1/{name}.jpg')
    else:
        img = image

    #############################################################################
    # Canny
    image_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(image_gray, (5, 5), 0)
    # closed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, np.ones([5,5]))
    edges = cv.Canny(blurred, 50, 110)
    edges = cv.morphologyEx(edges, cv.MORPH_DILATE, np.ones([5, 5]))
    #############################################################################

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)

    # List of contours (np arrays) where pos 0 is the biggest in length
    reversed_sorted_contours = sorted(contours, key=len, reverse=True)

    # cv2.drawContours(img_c, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    mask = np.zeros(shape=img.shape[:2])

    # Consider largest contour to be the first painting
    first_contour = reversed_sorted_contours[0]
    second_contour = []
    for contour in reversed_sorted_contours:
        if not contours_overlap(first_contour, contour):
            second_contour = contour
            break

    list_of_painting_coordinates = []

    # First painting
    Ax, Ay, Aw, Ah = cv.boundingRect(first_contour)
    list_of_painting_coordinates.append([Ax, Ay, Ax+Aw, Ay+Ah])
    mask[Ay:Ay+Ah, Ax:Ax+Aw] = 1

    # Second painting (not always there is a second painting)
    if len(second_contour) > 0:
        Bx, By, Bw, Bh = cv.boundingRect(second_contour)
        if (Bw * Bh) > ((Aw * Ah) / 100):  # area of the second painting is
            list_of_painting_coordinates.append([Bx, By, Bx+Bw, By+Bh])
            mask[By:By+Bh, Bx:Bx+Bw] = 1

    if save:
        if not os.path.exists(f'../datasets/masks_extracted/canny/'):
            os.makedirs(f'../datasets/masks_extracted/canny/')
        cv.imwrite(f'../datasets/masks_extracted/canny/{name}.png', mask)

    if generate_measures:
        return get_measures(image, mask)
    else:
        # List element: [x, y, x+w, y+h]
        return np.uint8(mask), list_of_painting_coordinates

## Utils
def rotate_rect(rect, angle):
    centroid = np.mean(np.array(rect), 0)
    # print('Centroid', centroid)
    # print('Angle', angle)
    # print('Rect', rect[0])
    centroid = tuple(centroid)
    rot_mat = cv.getRotationMatrix2D(centroid, angle, 1.0)
    for i in range(len(rect)):
        rect[i] = np.int16(rot_mat.dot(np.array(list(rect[i])+[1])))
    return rect


def rotate_image(image, angle, centroid=None):
    shape = image.shape[:2]
    image_center = tuple(np.array(shape[1::-1]) / 2) if centroid is None else centroid
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image.copy(), rot_mat, shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def rect4_to_rect2(rect):
    '''
    Gets a rect defined by its four corners and returns only top left and bottom right points
    in the format (tly,tlx, bry, brx)
    '''
    sorted_points = sorted(rect, key=lambda x: x[0]+x[1])
    tl = sorted_points[0]
    br = sorted_points[-1]

    return (*tl[::-1], *br[::-1])

def get_rrect_area(rect, angle):
    '''
    Gets a rotated rect in the format angle, [pt1,pt2,pt3,p4] and returns area
    '''
    # Rotate rect
    straight_rect = rotate_rect(rect.copy(), angle)

    # Get 2 opposite corners
    r = rect4_to_rect2(straight_rect.copy())

    # Gets area
    return (r[2]-r[0])*(r[3]-r[1])

def draw_rrect(im, pts, color=(0, 255, 0)):

    centroid = np.int16(np.mean(np.array(pts), 0))

    # Draw lines
    im = cv.line(im, tuple(pts[0]), tuple(pts[1]), color, 5)
    im = cv.line(im, tuple(pts[1]), tuple(pts[2]), color, 5)
    im = cv.line(im, tuple(pts[2]), tuple(pts[3]), color, 5)
    im = cv.line(im, tuple(pts[3]), tuple(pts[0]), color, 5)

    # Draw vertices
    for i in range(4):
        im = cv.circle(im, tuple(pts[i]), 2, (0,0, 255), -1)

    # Draw centroid
    im = cv.circle(im, tuple(centroid), 3, (0,0, 255), -1)

    return im

def method_canny_multiple_paintings_rot(image):

    img = image

    #############################################################################
    # Canny
    image_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(image_gray, (5, 5), 0)
    # closed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, np.ones([5,5]))
    edges = cv.Canny(blurred, 40, 120)
    edges = cv.morphologyEx(edges, cv.MORPH_DILATE, np.ones([5, 5]))
    #############################################################################

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)

    # List of contours (np arrays) where pos 0 is the biggest in length
    reversed_sorted_contours = sorted(contours, key=len, reverse=True)

    # Consider largest contour to be the first painting
    first_contour = reversed_sorted_contours[0]

    list_of_painting_coordinates = [] # now format is [angle, [pt1,pt2,pt3,pt4]]

    # First painting
    # TODO: min(Aw,Ah) > max(Aw,Ah)/5

    first_rect = cv.minAreaRect(first_contour)
    save_angle = -first_rect[2] if abs(first_rect[2]) <= 45 else 180 + first_rect[2]
    save_angle = save_angle if save_angle <= 45 else save_angle + 90
    save_angle = int(save_angle) % 180



    first_box = cv.boxPoints(first_rect)
    first_box = np.int0(first_box) # The four corners
    first_angle = first_rect[2] if abs(first_rect[2]) <= 45 else 90 + first_rect[2]
    first_centroid = tuple(np.mean(np.array(first_box), 0))

    '''
    # lower points
    lower_points = sorted(first_box, key=lambda x: x[1], reverse=True)[:2]
    rho1, theta1 = gu_line_polar_params_from_points(lower_points[0], lower_points[1])

    '''

    #TODO Check angles and point order is correct
    first_pkl = [save_angle, first_box]


    # Display stuff
    # print(first_angle)
    # edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # draw_edges = draw_rrect(edges.copy(), first_box)
    # cv.imshow('ahi', draw_edges)
    # cv.waitKey()

    # edges = cv.drawContours(edges,[first_box],0,(0,0,255),1)
    # rot_edges = rotate_image(edges.copy(), first_angle, centroid=first_centroid)
    # rot_edges = draw_rrect(rot_edges.copy(), rotate_rect(first_box.copy(), first_angle))
    # cv.imshow('canny', cv.resize(rot_edges.copy(), (500, 500*rot_edges.shape[0]//rot_edges.shape[1])))
    # cv.waitKey(0)

    # print(first_box.copy())
    # print(rotate_rect(first_box.copy(), first_angle))
    # print(rect4_to_rect2(rotate_rect(first_box.copy(), first_angle)))

    list_of_painting_coordinates.append([first_angle, first_centroid, rect4_to_rect2(rotate_rect(first_box.copy(), first_angle)), first_pkl])

    # Second painting (not always there is a second painting)
    # print("="*25)
    edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    for contour in reversed_sorted_contours:
        hyp_rect = cv.minAreaRect(contour)
        inter = cv.rotatedRectangleIntersection(first_rect, hyp_rect)

        hyp_box = cv.boxPoints(hyp_rect)
        hyp_box = np.int0(hyp_box) # The four corners
        hyp_angle = hyp_rect[2] if abs(hyp_rect[2]) <= 45 else 90 + hyp_rect[2]
        hyp_centroid = tuple(np.mean(np.array(hyp_box), 0))

        save_angle = -hyp_rect[2] if hyp_rect[1][0] > hyp_rect[1][1] else hyp_rect[2] + 180
        save_angle = save_angle if save_angle <= 45 else save_angle + 90
        save_angle = int(save_angle) % 180

        hyp_pkl = [save_angle, hyp_box]
        # if inter[0] == cv.INTERSECT_NONE:
        #     print('No intersection')
        # elif inter[0] == cv.INTERSECT_PARTIAL:
        #     print('Partial intersection')
        #     print(inter[1])
        # elif inter[0] == cv.INTERSECT_FULL:
        #     print('Full intersection')
        # continue

        if inter[0] == cv.INTERSECT_NONE: #TODO: Maybe try partial intersection with small area
            if get_rrect_area(hyp_box, hyp_angle) > get_rrect_area(first_box, first_angle) / 4: #TODO and min(Bw,Bh) > max(Bw,Bh)/5:  # area of the second painting is
                # draw_ed = draw_rrect(edges.copy(), hyp_box)
                # cv.imshow('after', cv.resize(draw_ed, (500, 500*edges.shape[0]//edges.shape[1])))
                # rot_edges = rotate_image(edges.copy(), hyp_angle, centroid=hyp_centroid)
                # rot_edges = draw_rrect(rot_edges.copy(), rotate_rect(hyp_box.copy(), hyp_angle))
                # cv.imshow('canny sec', cv.resize(rot_edges, (500, 500*rot_edges.shape[0]//rot_edges.shape[1])))
                # cv.waitKey(0)
                
                list_of_painting_coordinates.append([hyp_angle, hyp_centroid, rect4_to_rect2(rotate_rect(hyp_box, hyp_angle)), hyp_pkl])
            else:
                break
    
    # input('Press any key to continue...')

    return 0, list_of_painting_coordinates


def method_canny_multiple_paintings(image, save=False, generate_measures=False):

    if generate_measures:
        name = image
        img = cv.imread(f'../datasets/qsd2_w1/{name}.jpg')
    else:
        img = image

    #############################################################################
    # Canny
    image_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(image_gray, (5, 5), 0)
    # closed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, np.ones([5,5]))
    edges = cv.Canny(blurred, 40, 120)
    edges = cv.morphologyEx(edges, cv.MORPH_DILATE, np.ones([5, 5]))
    #############################################################################

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)

    # List of contours (np arrays) where pos 0 is the biggest in length
    reversed_sorted_contours = sorted(contours, key=len, reverse=True)

    # cv2.drawContours(img_c, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    mask = np.zeros(shape=img.shape[:2])

    # Consider largest contour to be the first painting
    first_contour = reversed_sorted_contours[0]

    list_of_painting_coordinates = []

    # First painting
    # TODO: min(Aw,Ah) > max(Aw,Ah)/5
    
    Ax, Ay, Aw, Ah = cv.boundingRect(first_contour)
    list_of_painting_coordinates.append([Ax, Ay, Ax+Aw, Ay+Ah])
    mask[Ay:Ay+Ah, Ax:Ax+Aw] = 1
    image_bb = image.copy()
    cv.rectangle(image_bb, (Ax, Ay), (Ax + Aw, Ay + Ah), (0, 255, 0), 15)

    # Second painting (not always there is a second painting)
    # print("="*25)
    for contour in reversed_sorted_contours:
        if not contours_overlap(first_contour, contour):
            Bx, By, Bw, Bh = cv.boundingRect(contour)
            if (Bw * Bh) > ((Aw * Ah) / 5) and min(Bw,Bh) > max(Bw,Bh)/5:  # area of the second painting is
                # print("Width: ", Bw)
                # print("Height: ", Bh)
                list_of_painting_coordinates.append([Bx, By, Bx+Bw, By+Bh])
                mask[By:By+Bh, Bx:Bx+Bw] = 1
                # cv.rectangle(image_bb, (Bx, By), (Bx + Bw, By + Bh), (0, 255, 0), 15)
            else:
                break
    # plt.imshow(image_bb)
    # plt.show()
    if save:
        if not os.path.exists(f'../datasets/masks_extracted/canny/'):
            os.makedirs(f'../datasets/masks_extracted/canny/')
        cv.imwrite(f'../datasets/masks_extracted/canny/{name}.png', mask)

    if generate_measures:
        return get_measures(image, mask)
    else:
        # List element: [x, y, x+w, y+h]
        return np.uint8(mask), list_of_painting_coordinates


def method_canny(image, save=False, generate_measures=False):
    """
    Calculate background limits regarding painting by detecting lines belonging to painting's frame.
    Assumes a smooth background.
    :param img:  image (BGR)
    :return: binary mask
    """

    if generate_measures:
        name = image
        img = cv.imread(f'../datasets/qsd2_w1/{name}.jpg')
    else:
        img = image

    image_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    mask = np.zeros_like(image_gray).astype(np.int32)

    blurred = cv.GaussianBlur(image_gray, (5, 5), 0)
    canny = cv.Canny(blurred, 50, 150)

    sum_col_values = canny.sum(axis=0)
    sum_row_values = canny.sum(axis=1)

    upper_frame = np.nonzero(sum_row_values)[0][0]
    lower_frame = (canny.shape[0]-1) - np.nonzero(sum_row_values[::-1])[0][0]
    left_frame = np.nonzero(sum_col_values)[0][0]
    right_frame = (canny.shape[1]-1) - np.nonzero(sum_col_values[::-1])[0][0]

    mask[upper_frame:lower_frame, left_frame:right_frame] = 1  # pixels corresponding to detected painting area

    if save:
        if not os.path.exists(f'../datasets/masks_extracted/canny/'):
            os.makedirs(f'../datasets/masks_extracted/canny/')
        cv.imwrite(f'../datasets/masks_extracted/canny/{name}.png', mask)

    if generate_measures:
        return get_measures(image, mask)
    else:
        return [(mask.astype(np.uint8), (upper_frame, left_frame, lower_frame, right_frame))]


def get_all_methods_per_photo(im, display, save=False):
    """
    Return a dictionary with all available measures. Keys are in example:
    * 'msc': method_similar_channels
    * 'mst': method_colorspace_thresholding(image, range x[a,b], range y[c,d], range z[e,f], colorspace)
    """

    measures = {
                # 'msc': method_similar_channels(im, 30, save=save, generate_measures=True),
                # 'mst_bgr': method_colorspace_threshold(im, [124, 255], [0, 255], [0, 255], 'bgr', save=save,
                #                                    generate_measures=True),
                # 'mst_hsv': method_colorspace_threshold(im, [0, 255], [0, 255], [140, 255], 'hsv', save=save,
                #                                    generate_measures=True),
                # 'msk_bgr': method_mostcommon_color_kmeans(im, 5, 30, colorspace='bgr', save=save,
                #                                           generate_measures=True),
                # 'msk_hsv': method_mostcommon_color_kmeans(im, 5, 10, colorspace='hsv', save=save,
                #                                           generate_measures=True),
                # 'canny': method_canny(im, save=save, generate_measures=True),
                # 'watershed': method_watershed(im, save=save, generate_measures=True),
                'hsv_morph': hsv_thresh_method(im)
                }
    # methods returning masks and should return measures
    if display:
        for k in measures.items():
            print(k)

    return measures


def get_all_measures_all_photos(save=False):
    """
    calculates all measures for all photos and saving masks in folders if necessary
    Methods to run are defined in get_all_methods_per_photo

    csv file is saved in the folde above your project

    """
    files_img = glob.glob('../datasets/qsd2_w1/*.png')

    all_measures = []

    for index, image in enumerate(files_img):
        image = image[-9:-4]
        measure = get_all_methods_per_photo(image, display=True, save=save)
        all_measures.append(measure)

    with open('../background_removal_all_methods_all_photos.csv', 'a') as f:
        all_rows = []
        header = 'method' + ',name' + ',precision' + ',recall' + ',F1_measure'
        all_rows.append(header)
        for item in all_measures:
            for k, v in item.items():
                row = "\n" + k
                for i in v.values():
                    row = row + "," + str(i)
                # print(row)
                all_rows.append(row)

        f.writelines(all_rows)

    return all_measures


def main(name, display, save=False):
    if not os.path.exists('../datasets/masks_extracted/'):
        os.makedirs('../datasets/masks_extracted/')

    if not os.path.exists('pkl_data/masks_extracted/'):
        os.makedirs('pkl_data/masks_extracted/')

    get_all_methods_per_photo(name, display, save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, type=str, help='Name of the imagefile')
    parser.add_argument('-d', '--display', required=False, type=bool, default=True, help='display measures')
    parser.add_argument('-s', '--save', required=False, type=bool, default=False, help='display measures')
    args = parser.parse_args()

    main(args.name, args.display, args.save)

