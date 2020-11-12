import pickle as pkl
import numpy as np
import cv2


def bbox_iou(bboxA, bboxB):
    '''
    compute the intersection over union of two bboxes

    Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    indicate top-left and bottom-right corners of the bbox respectively.
    '''
    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou, (xA, yA, xB, yB)


def gu_line_polar_intersection(rho1, theta1, rho2, theta2):
    # Angles in radians
    #if np.sin(theta1-theta2) < 1.e-6:
    #    return
    y = (rho1*np.cos(theta2)-rho2*np.cos(theta1))/np.sin(theta1-theta2)
    x = (rho1 - y*np.sin(theta1))/np.cos(theta1)
    return (x,y)


def gu_line_polar_params_from_points(p1, p2):
    #theta = np.arctan((p1[0]-p2[0])/(p2[1]-p1[1]))
    theta = np.arctan2(p1[0]-p2[0], p2[1]-p1[1])
    rho   = p1[0]*np.cos(theta) + p1[1]*np.sin(theta)

    return (rho, theta)


def angular_error_boxes(box1, box2):
    # Compute angle 1
    # 1: take the two points in the lower  part and compute angle
    lower_points = sorted(box1, key=lambda x: x[1], reverse=True)[:2]
    rho1, theta1 = gu_line_polar_params_from_points(lower_points[0], lower_points[1])

    # Compute angle 2
    # 2: take the two points in the lower  part and compute angle
    lower_points = sorted(box2, key=lambda x: x[1], reverse=True)[:2]
    rho2, theta2 = gu_line_polar_params_from_points(lower_points[0], lower_points[1])

    print (theta1, theta2)
    return abs(theta1-theta2)


def angular_error_box_angle (gt_box, hyp_angle):
    # Compute angle 1
    # 1: take the two points in the lower  part and compute angle
    lower_points = sorted(gt_box, key=lambda x: x[1], reverse=True)[:2]
    rho1, theta1 = gu_line_polar_params_from_points(lower_points[0], lower_points[1])

    if theta1 > 0:
        gt_angle = ((3.0*np.pi)/2 - theta1) * (180.0/np.pi)
    else:
        gt_angle = ((3.0*np.pi)/2 + theta1) * (180.0/np.pi)

    print (theta1*(180.0/np.pi), gt_angle, hyp_angle)
    return abs(gt_angle - hyp_angle)


def get_xy(event, x, y, *etc):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def draw_rrect(im, pts, color=(0, 255,0)):
    # pts tuple with tlx, tly, brx, bry
    # for i in range(len(pts)):
    #     pts[i] = pts[i][::-1]
    centroid = np.int16(np.mean(np.array(pts), 0))

    im = cv2.line(im, tuple(pts[0]), tuple(pts[1]), color, 2)
    im = cv2.line(im, tuple(pts[1]), tuple(pts[2]), color, 2)
    im = cv2.line(im, tuple(pts[2]), tuple(pts[3]), color, 2)
    im = cv2.line(im, tuple(pts[3]), tuple(pts[0]), color, 2)
    im = cv2.circle(im, tuple(pts[0]), 2, (0,0, 255), -1)
    im = cv2.circle(im, tuple(pts[1]), 2, (0,0, 255), -1)
    im = cv2.circle(im, tuple(pts[2]), 2, (0,0, 255), -1)
    im = cv2.circle(im, tuple(pts[3]), 2, (0,0, 255), -1)
    im = cv2.circle(im, tuple(centroid), 3, (0,0, 255), -1)


    return im

def rotate_rect(rect, angle):
    centroid = np.mean(np.array(rect), 0)
    # print('Centroid', centroid)
    # print('Angle', angle)
    # print('Rect', rect[0])
    centroid = tuple(centroid)
    rot_mat = cv2.getRotationMatrix2D(centroid, angle, 1.0)
    for i in range(len(rect)):
        rect[i] = np.int16(rot_mat.dot(np.array(list(rect[i])+[1])))
    return rect

def rotate_image(image, angle):
    shape = image.shape[:2]
    image_center = tuple(np.array(shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, shape[1::-1], flags=cv2.INTER_LINEAR)
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

def rbox_iou(box1, box2):
    '''
    Gets two rotated boxes and returns the IoU between them.
    Input format:
        box<n> = [alpha, [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]]
        Where alpha is the box's angle in radians, and x<n>,y<n> the four vertices of the box
    '''
    # Rotate boxes
    rect1 = rotate_rect(box1[1].copy(), -box1[0])
    rect2 = rotate_rect(box2[1].copy(), -box2[0])

    # Convert rects to (tlx,tly, brx, bry) format
    rect1 = rect4_to_rect2(rect1)
    rect2 = rect4_to_rect2(rect2)

    # Used while debugging
    ima = np.zeros((777, 777, 3), dtype=np.uint8)
    ima = cv2.rectangle(ima, (rect1[0], rect1[1]), (rect1[2], rect1[3]), (0,0,255))
    ima = cv2.rectangle(ima, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0,255,255))
    cv2.imshow('ima', ima)

    # Get iou
    return bbox_iou(rect1, rect2)

with open('../datasets/qsd1_w5/frames.pkl', 'rb') as f:
    gt_frames = pkl.load(f)

# with open('pkl_data/frames.pkl', 'rb') as f:
#     hyp_frames = pkl.load(f)

sample_rect = gt_frames[5][0]
r_angle = 130
rotated_r = rotate_rect(sample_rect[1].copy(), r_angle)

print('Mine:', rbox_iou(sample_rect, [r_angle, rotated_r])[0])
his_iou, his_rect = bbox_iou(rect4_to_rect2(sample_rect[1]), rect4_to_rect2(rotated_r))
print('His:', his_iou)

im = np.zeros((777, 777, 3), dtype=np.uint8)
im = draw_rrect(im, sample_rect[1])
im = draw_rrect(im, rotated_r, (255, 0, 0))
im = cv2.rectangle(im, (his_rect[0], his_rect[1]), (his_rect[2], his_rect[3]), (0,0,255))
cv2.imshow('rct', im)
cv2.waitKey(0)

quit()

sum_iou = 0
n = 0
for image in gt_frames:
    for painting in image:
        alpha = painting[0]
        vertices = painting[1]
        # get iou
        # print(alpha, vertices)
        n += 1

# print(f'Mean IoU is {sum_iou/n}')


img_idx = 8
img_path = '../datasets/qsd1_w5/{:05d}.jpg'.format(img_idx)
im = cv2.imread(img_path)
if im is None:
    print('Error reading image', img_path)
    quit()

i = 0
while True:
    sample_rrect = gt_frames[i][0]
    print(f' Image: {img_path}\n Frame: {i} \n Image size: {im.shape}\n Angle: {sample_rrect[0]} \n Vertices: {sample_rrect[1]}\n')
    # tlx, tly, brx, bry = sample_rrect[1]

    im_draw = draw_rrect(im.copy(), sample_rrect[1])
    im_rot = rotate_image(im, -sample_rrect[0])

    cv2.namedWindow('im')
    cv2.setMouseCallback('im', get_xy)

    cv2.imshow('im', im_draw)
    cv2.imshow('rotated im', im_rot)

    k = cv2.waitKey(0)

    if k is ord('q'):
        break
    elif k is ord('d'):
        i+=1
    elif k is ord('a'):
        i -= 1
