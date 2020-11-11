import pickle as pkl 

with open('frames.pkl', 'rb') as f:
    frames = pkl.load(f)


# format of frames.pkl is 
# [ #all images
#   [ #image 1
#     [alpha1, # painting 1
#       [ (px1, py1), (px2, py2), (px3, py3), (px4, py4)]
#      ], 
#      ..., 
#     [alphaN, # painting N
#       [(px1, py1), (px2, py2), (px3, py3), (px4, py4)]
#     ]
#   ],
#   ...,
#   [ #image K
#     [alpha1, # painting 1
#       [ (px1, py1), (px2, py2), (px3, py3), (px4, py4)]
#      ], 
#      ..., 
#     [alphaN, # painting N
#       [(px1, py1), (px2, py2), (px3, py3), (px4, py4)]
#     ]
#   ]
# ]

sum_iou = 0
n = 0
for image in frames:
    for painting in image:
        alpha = painting[0]
        vertices = painting[1]
        # get iou
        n += 1

print(f'Mean IoU is {sum_iou/n}')
