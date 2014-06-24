'''
Created on 22.4.2014

@author: Michal
'''

import numpy as np
import cv2
from cv2 import cv
import pickle
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.morphology
import skimage.measure
import config

plt.gray()
MIN_MATCH_COUNT = config.min_count_of_matches

im1 = config.image_kinect
im2 = config.image_projector

img1 = cv2.imread(im1,0)      # queryImage
img2 = cv2.imread(im2,0) # trainImage

img1 = skimage.color.rgb2gray(img1)
img2 = skimage.color.rgb2gray(img2)

# Initiate SIFT detector
sift = cv2.SIFT()
 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)         
         
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    print M
       
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None  

print len(good)

kalib_params = M
with open('matice_kal2', 'wb') as f:
    pickle.dump(kalib_params,f)

import kalibrace2

plt.figure(0)
plt.imshow(img1)
plt.hold(True)
for i in range(0,len(src_pts)-1):
    plt.plot(src_pts[i][0][0],src_pts[i][0][1],'r+')
plt.title('Vyznamne body - kinect')    
 
 
plt.figure(1)
plt.imshow(img2)
plt.hold(True)
for i in range(0,len(src_pts)-1):
    plt.plot(dst_pts[i][0][0],dst_pts[i][0][1],'r+')
plt.title('Vyznamne body - projektor') 


# src = src_pts[3][0] #bod pro transformaci
# plt.figure(2)
# plt.imshow(img1)
# plt.hold(True)
# plt.plot(src[0],src[1],'g+')


# plt.figure(3)
# plt.imshow(img2)
# plt.hold(True)
# for src_pt in src_pts:
#     src = src_pt[0]
#     telotr = kalibrace2.projekce(src, kalib_params, mode = 'ransac')
#     plt.plot(telotr[0],telotr[1],'g+')
# plt.title('Vyznamne body - projektor - po transformaci')


plt.figure(3)
plt.imshow(img2)
plt.hold(True)
for src_pt in src_pts:
    src = src_pt[0]
    pt = np.float32([ [src[0],src[1]]]).reshape(-1,1,2)
    ip = cv2.perspectiveTransform(pt, M)
    proj_point = ip[0,0]
    plt.plot(proj_point[0],proj_point[1],'g+')
plt.title('Vyznamne body - kinect - po transformaci')
plt.show()
