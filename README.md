# Team6
Team 6's repository 

Folder structure for running:
..
├── datasets
│   ├── BBDD
│   ├── qsd1_w1
│   └── qsd2_w1
└── Team6
    ├── evaluation
        ├── pkl_data
    
run code:

1. create db with descriptors for BBDD by running once

index_db.py

2. to get results run the cbir to get list of K most similar paintings

cbir.py -i 00001 -d 'bgr_concat_hist' -m 'corr'

args:
-i {name of image without extension}
-d {descriptor name from descriptor_lib.py} 
-m {metric from distance_metrics_lib.py}

example run:
cbir.py -i 00001 -d 'bgr_concat_hist' -m 'corr'

example result:
[120, 201, 304]

3. to get a .csv with map@k run the retrieval_evaluation.py

example run:

python retrieval_evaluation.py -q qsd1_w1 -d 'bgr_concat_hist' -m 'corr' -k 5

args:
-q {name of the query set - it has to be inside datasets directory}
-d {descriptor name from descriptor_lib.py} 
-m {metric from distance_metrics_lib.py}
-k {number of most similar paintings to find}

3. to get results for background removal for an image for all methods run background_removal.py

example run:

python background_removal.py -i 00003 -d True -s True

args:
-i {name of the image without extension}
-d {display of measures}
-s {save the mask as an image}

example result:
('msc', {'name': '00004', 'precision': 0.8221164520426287, 'recall': 0.9300189943958683, 'F1_measure': 0.872745217910096})
('mst', {'name': '00004', 'precision': 0.8670100260953166, 'recall': 0.9909580396527636, 'F1_measure': 0.9248496480188699})
