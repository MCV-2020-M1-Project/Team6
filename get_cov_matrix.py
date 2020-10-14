import os
import pickle  as pkl
import numpy as np
import cv2

#Read descriptors of the museum db from .pkl
path = ['pkl_data','bd_descriptors.pkl'] #for making the path system independent
db_descript_list = []

with open(os.path.join(*path), 'rb') as dbfile:
    db_descript_list = pkl.load(dbfile)

all_hist = np.zeros((256, 287))
for i, h in enumerate(db_descript_list):
    a = h['gray_hist'].copy()
    cv2.normalize(a, a, norm_type=cv2.NORM_L2, alpha=1.)

    all_hist[:, i] = np.transpose(np.array(a))

M = np.cov(all_hist)
M_inv = np.linalg.inv(M)

with open('gray_sim_mat.pkl', 'wb') as f:
    pkl.dump(M_inv, f)


    