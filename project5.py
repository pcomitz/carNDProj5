# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 11:53:05 2017

CarND Project 5
This is the main python source file 
for the Udacity CarND Vehicle Detection and Tracking 
Project (Project 5) of semester 1. 

Date of submission: 9/25/2017

@author: pcomitz

Typical Test result
-------------------
Output 9/23/2017
len(cars) is: 8792
cars example image: ../../../../../p5_train/vehicles/GTI_Far\image0210.png
len(noncars) is: 8968
notcars example image: ../../../../../p5_train/non-vehicles/Extras\extra1039.png
y:\Anaconda3\envs\carnd-term1\lib\site-packages\skimage\feature\_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
  'be changed to `L2-Hys` in v0.15', skimage_deprecation)
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
16.34 Seconds to train SVC...
Test Accuracy of SVC =  0.9866


RESUBMIT WITH HEATMAP MODS

"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import random
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
#from lesson_functions import *
from lesson_functions_34 import *
from multipleDetectionsAndFalsePositives_37 import *

#add heatmap history
from collections import deque
global heatmap_history
heatmap_history = deque(maxlen=15)

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
 

def process_image(img):
    
    # parameters
    # changed hist to 384
    color_space = 'YCrCb'
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 384    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [400, 700] # Min and max in y to search in slide_window()
    x_start_stop = [400,None] # changed to  200, 1280 then changed back  
 
    # per forum, copy image that is passed in
    # perform processing on copy
    # draw on original 
    copy = np.copy(img)
    image = copy.astype(np.float32)/255
   
    # returns list of windows to search
    # changed xy_window to 64,64 from 96,96, trying 32, 32
    # xy_overlap tp 0.75.0.75 from -0.5,0.5
    # made it very slow and it made it worse 
    # back to 0.5, -0.5 and 64, 64
   
    # returns list of windows to search
    windows = slide_window(image, x_start_stop=[400, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    # returns lits of windows that contain detections 
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       


    #heat map and false positives
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    
    #add heatmap history
    heatmap_history.append(heat)
    
    # combined is an image size array - sum of 
    # all heat arrays 
    combined = np.sum(heatmap_history, axis = 0)
    
    # Apply threshold to help remove false positives
    # heat = apply_threshold(heat,1)
    heat = apply_threshold(combined, 4)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # draw on original image
    draw_img = draw_labeled_bboxes(img, labels)
    
    return draw_img

###########
# B E G I N
###########

# get the car training images (strings with path to images)
# path = '../../../../../p5_train/'
path = './'

images = []
"""
images = glob.glob('../../../../../p5_train/vehicles/GTI_Far/*.png')
images.extend(glob.glob('../../../../../p5_train/vehicles/GTI_Left/*.png'))
images.extend(glob.glob('../../../../../p5_train/vehicles/GTI_MiddleClose/*.png'))
images.extend(glob.glob('../../../../../p5_train/vehicles/GTI_Right/*.png'))
images.extend(glob.glob('../../../../../p5_train/vehicles/KITTI_Extracted/*.png')) 
"""      

#laptop
images = glob.glob('vehicles/GTI_Far/*.png')
images.extend(glob.glob(path+'vehicles/vehicles/GTI_Left/*.png'))
images.extend(glob.glob(path+'vehicles/vehicles/GTI_MiddleClose/*.png'))
images.extend(glob.glob(path+'vehicles/vehicles/GTI_Right/*.png'))
images.extend(glob.glob(path+'vehicles/vehicles/KITTI_Extracted/*.png')) 

      
cars = []
for image in images:
        cars.append(image)
        
print("len(cars) is:", len(cars))     
random.randint(0,len(cars)-1)
image = mpimg.imread(cars[random.randint(0,len(cars)-1)])
print("cars example image:", cars[random.randint(0,len(cars)-1)])
plt.imshow(image)   

#get the not cars training images
#nonimages = []
#wpath = 'non-vehicles/Extras/*.png'
nimages = glob.glob('non-vehicles/non-vehicles/Extras/*.png')
nimages.extend(glob.glob('non-vehicles/non-vehicles/GTI/*.png'))

notcars = []
for image in nimages:
        notcars.append(image)

print("len(noncars) is:", len(notcars))    
       
random.randint(0,len(notcars)-1)
image = mpimg.imread(notcars[random.randint(0,len(notcars)-1)])
print("notcars example image:", notcars[random.randint(0,len(notcars)-1)])
#plt.imshow(image)   

#phc
sample_size = 7900
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]  


### Tweak these parameters and see how the results change.
#color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# changed to YCrCb
# hist to 384 from 16
color_space = 'YCrCb'
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 384    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [300, 700] # Min and max in y to search in slide_window()
# x_start_stop = [200,1280] 

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


#
# Start reading the video here
#
proj5_output = 'proj5.mp4'
clip1 = VideoFileClip('project_video.mp4')
proj5_clip = clip1.fl_image(process_image) 
proj5_clip.write_videofile(proj5_output, audio=False)



