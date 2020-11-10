import cv2 
import numpy as np

from libs import box_retrieval as bbox 
from libs import background_removal as bg 
from libs import text_retrieval as txt
from libs import distance_metrics as dist

def sort_rects_lrtb(rect_list):
    '''
    Sorts rect lists from left to right and top to bottom
    '''
    return sorted(rect_list, key = lambda x: (x[0], x[1]))


test_set = 'qsd1_w4'
text_gt = []
for i in range(30):
    txt_path = '../datasets/{}/{:05d}.txt'.format(test_set, i)
    print(txt_path)
    with open(txt_path, 'r', encoding='ISO-8859-1') as fp:
        t = fp.read()
        t = t.splitlines()
        for line in t:
            text_info = line.strip()[1:-1].replace("'",'').split(',')
            text_info = tuple([line.strip() for line in text_info])

            print(text_info)
            text_gt.append(text_info)


methods = {'old': bbox.get_boxes_old, 'new': bbox.get_boxes}
results = []

for i in range(30):
    img_path = '../datasets/{}/{:05d}.jpg'.format(test_set, i)

    im = cv2.imread(img_path, 1)

    new_item = {}
    for k, f in methods.items():
        
        bbox = []
        masks = bg.method_canny_multiple_paintings(im.copy())[1]
        masks = sort_rects_lrtb(masks)
        paintings = []
        for mask in masks:
            if abs(mask[1] - mask[3]) < 50: # wrong crop
                continue
            v1 = mask[:2]
            v2 = mask[2:]
            paintings.append(im[v1[1]:v2[1], v1[0]:v2[0]])

        for paint in paintings:
            bbox.append(f(paint))

        mask = f(im)
        if isinstance(mask, tuple):
            mask = mask[1]

        if mask is None:
            new_item[k] = ''
            continue
        a = np.where(mask > 0)
        pts = [(i, j) for i,j in zip(*a)]

        if len(pts) == 0:
            new_item[k] = ''
            continue

        masked_img = im[pts[0][0]:pts[-1][0], pts[0][1]:pts[-1][1]]
        new_item[k] = txt.get_text(masked_img)

    results.append(new_item)

# Save results
for i, r in enumerate(results):
    print('ocr_results/{:05d}.txt'.format(i))
    print(r['new'])


results_error = {}
for k in results[0]:
    results_error[k] = 0

for gt, r in zip(text_gt, results):
    for k in r:
        results_error[k] += dist.gestalt(gt[0], r[k])

print(results_error)

