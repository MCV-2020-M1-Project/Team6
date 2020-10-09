#!/usr/bin/env python
# coding: utf-8

# ## M1 Project
# ### Task 3: Compute similarities for each image in QS1

# ### Step 1: Read datasets

# In[21]:


from matplotlib import pyplot as plt
import operator
import numpy as np
import cv2
import os 
import glob 


# In[3]:


# Museum database
ddbb_dir = "qsd1_w1"
im_dirs = sorted(glob.glob(ddbb_dir+'/*jpg'))
original_images = []
for im_dir in im_dirs:
    #fname = os.path.splitext(os.path.basename(im_dir))[0] # No need to get filenames since the list is sorted
    original_images.append(cv2.cvtColor(cv2.imread(im_dir), cv2.COLOR_BGR2RGB))

# Query images
query_dir = "qsd1_w1"
im_dirs = sorted(glob.glob(ddbb_dir+'/*jpg'))
query_images = []
for im_dir in im_dirs:
    #fname = os.path.splitext(os.path.basename(im_dir))[0] # No need to get filenames since the list is sorted
    query_images.append(cv2.cvtColor(cv2.imread(im_dir), cv2.COLOR_BGR2RGB))


# In[4]:


idx = 40
plt.title(str(idx).zfill(5)+'.jpg')
plt.imshow(original_images[idx])
plt.show()


# ### Step 2: Compute histograms, these will be used as a first approach for measuring similarity
# OpenCV computes Histograms with the function calcHist(). The parameters are the following:
# - Image: our original image
# - Channels: 3 channels, RGB
# - mask: used for partial histograms, set to None
# - histSize: 256, full scale for every channel
# - range: 0-255 for every channel
# 
# Note: this step can be made when first reading the dataset in step 1    

# In[5]:


hist_dict = {}
for idx,img in enumerate(original_images):
    hist = cv2.calcHist([img], [0, 1, 2], None, [256,256,256], [0, 256, 0, 256, 0, 256])
    hist_dict[idx] = hist


# #### Graphical example of computed histograms:

# In[6]:


def plot_histogram(img):
    for chan,color in enumerate(['r','g','b']):
        hist = cv2.calcHist([img], [chan], None, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])    
    plt.show()


plt.imshow(original_images[40])
plt.show()
plot_histogram(original_images[40])


# ### Step 3: Given a query image, return the most similar k images from the original dataset
# 

# In[22]:


def get_histogram_top_k_similar(hist1, hist_dict, k=3):
    method = cv2.HISTCMP_CORREL
    '''
    if similarity_method == 'Chi-Squared':
        method = cv2.HISTCMP_CHISQR
    elif similarity_method == 'Intersection':
        cv2.HISTCMP_INTERSECT
    elif similarity_method == 'Hellinger':
        method = cv2.HISTCMP_BHATTACHARYYA
    '''
    
    distances_dict = {}
    for idx in hist_dict.keys():
        histogram = hist_dict[idx]
        distance = cv2.compareHist(hist1, hist_dict[idx], method)
        distances_dict[idx] = abs(distance)
    
    
    result = [key for key in sorted(distances_dict, key=distances_dict.get, reverse=True)[:k]]

    return result


# In[8]:


plt.imshow(query_images[0])
plt.show()
plot_histogram(query_images[0])


# #### Find the 3 most similar images for query image 1

# In[27]:


query_image = query_images[15]

hist_im_1 = cv2.calcHist([query_image], [0, 1, 2], None, [256,256,256], [0, 256, 0, 256, 0, 256])
top_matches = get_histogram_top_k_similar(hist_im_1, hist_dict, k=3)


# In[31]:


plt.title('Query Image')
plt.imshow(query_image)
plt.show()

for k,match in enumerate(top_matches):
    plt.title('Match number ' + str(k+1) + ', ' + str(match).zfill(5)+'.jpg')
    plt.axis('off')
    plt.imshow(original_images[match])
    plt.show()


# In[ ]:




