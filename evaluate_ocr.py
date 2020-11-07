import cv2 
import numpy as np

from libs import box_retrieval as bbox 
from libs import background_removal as bg 
from libs import text_retrieval as txt
from libs import distance_metrics as dist

test_set = 'qst1_w3'
# text_gt = []
# for i in range(30):
#     txt_path = '../datasets/{}/{:05d}.txt'.format(test_set, i)
#     print(txt_path)
#     with open(txt_path, 'r', encoding='ISO-8859-1') as fp:
#         t = fp.read()
#         text_info = t.strip()[1:-1].replace("'",'').split(',')
#         text_info = tuple([t.strip() for t in text_info])

#         # print(text_info)
#         text_gt.append(text_info)


methods = {'carmen': bbox.get_boxes, 'adam': bbox.filled_boxes}
results = []

for i in range(30):
    img_path = '../datasets/{}/{:05d}.jpg'.format(test_set, i)

    im = cv2.imread(img_path, 1)

    new_item = {}
    for k, f in methods.items():

        mask = f(im)
        if isinstance(mask, tuple):
            mask = mask[1]

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
    print(r['carmen'])


# results_error = {}
# for k in results[0]:
#     results_error[k] = 0

# for gt, r in zip(text_gt, results):
#     for k in r:
#         results_error[k] += dist.gestalt(gt[0], r[k])

# print(results_error)

