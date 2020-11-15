import pickle as pkl
import numpy as np
import cv2

# Their functions
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
    return iou


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

    # print (theta1, theta2, file=sys.stderr)
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

    # print (theta1*(180.0/np.pi), gt_angle, hyp_angle)
    return abs(gt_angle - hyp_angle)

# Our funcitons
## ui
def get_xy(event, x, y, *etc):
    '''
    Used for getting pixel coordinates in imshow, just for debugging
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def draw_rrect(im, pts, color=(0, 255, 0)):

    centroid = np.int16(np.mean(np.array(pts), 0))

    # Draw lines
    im = cv2.line(im, tuple(pts[0]), tuple(pts[1]), color, 2)
    im = cv2.line(im, tuple(pts[1]), tuple(pts[2]), color, 2)
    im = cv2.line(im, tuple(pts[2]), tuple(pts[3]), color, 2)
    im = cv2.line(im, tuple(pts[3]), tuple(pts[0]), color, 2)

    # Draw vertices
    for i in range(4):
        im = cv2.circle(im, tuple(pts[i]), 2, (0,0, 255), -1)

    # Draw centroid
    im = cv2.circle(im, tuple(centroid), 3, (0,0, 255), -1)

    return im

## Utils
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

## Actual stuff
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
    # ima = np.zeros((777, 777, 3), dtype=np.uint8)
    # ima = cv2.rectangle(ima, (rect1[0], rect1[1]), (rect1[2], rect1[3]), (0,0,255))
    # ima = cv2.rectangle(ima, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0,255,255))
    # cv2.imshow('ima', ima)

    # Get iou
    return bbox_iou(rect1, rect2)


def debug(gt_frames):
    '''
    Just for testing rotated iou was working
    '''
    sample_rect = gt_frames[5][0]
    r_angle = 130
    rotated_r = rotate_rect(sample_rect[1].copy(), r_angle)

    print('Mine:', rbox_iou(sample_rect, [r_angle, rotated_r]))
    his_iou = bbox_iou(rect4_to_rect2(sample_rect[1]), rect4_to_rect2(rotated_r))
    print('His:', his_iou)

    im = np.zeros((777, 777, 3), dtype=np.uint8)
    im = draw_rrect(im, sample_rect[1])
    im = draw_rrect(im, rotated_r, (255, 0, 0))

    cv2.imshow('rct', im)
    cv2.waitKey(0)

    quit()

def check_point_order(box):
    print(box)
    print(box.shape)
    x = np.max(box[:,0])+100
    y = np.max(box[:,1])+100
    ecran = np.zeros((y, x, 3))
    print(ecran.shape)
    for i, p in enumerate(box):
        ecran = cv2.putText(ecran, str(i), tuple(p), cv2.FONT_HERSHEY_COMPLEX, 4, (0,0,255))
    
    cv2.imshow('Points', cv2.resize(ecran, (500, 500*ecran.shape[0]//ecran.shape[1])))
    cv2.waitKey(0)

def get_rrect_props(rrect):
    angle = rrect[0]
    vertex = rrect[1]

    # RotatedRect rotated_rect = minAreaRect(contour);
    # float blob_angle_deg = rotated_rect.angle;
    # if (rotated_rect.size.width < rotated_rect.size.height) {
    # blob_angle_deg = 90 + blob_angle_deg;
    # }

def main():
    '''
    Gets the mean IoU and mean Angular Error. Our program's results should be
    in pkl_results/frames.pkl
    '''
    with open('../datasets/qsd1_w5/frames.pkl', 'rb') as f:
        gt_frames = pkl.load(f)

    with open('pkl_data/frames.pkl', 'rb') as f:
        hyp_frames = pkl.load(f)

    sum_iou = 0
    sum_ae = 0

    n = 0
    for gt_image, hyp_img in zip(gt_frames, hyp_frames):
        for gt_painting, hyp_painting in zip(gt_image, hyp_img):
            print('gt angle', gt_painting[0])
            print('hyp angle', hyp_painting[0], '\n')
            sum_iou += rbox_iou(gt_painting, hyp_painting)
            sum_ae += angular_error_boxes(gt_painting[1], hyp_painting[1])
            n += 1

    # TODO: divide by n or do the TP thing and divide only by those?
    print(f'Mean IoU is {sum_iou/n}') 
    print(f'Mean Angular Error is {sum_ae/n}')


if __name__ == '__main__':
    # debug()
    main()


# with open('pkl_data/harmony.pkl', 'wb') as f:
#     pkl.dump(harm_dict, f)
