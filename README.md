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
