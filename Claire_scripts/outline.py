#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:11:11 2020

@author: Claire
"""

from __future__ import division    
import cv2
import numpy as np

# import seaborn as sns


#%%
# Open Video

cap = cv2.VideoCapture('Reverse_888.mp4')

# Randomly select 25 frames

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=1)

# Store selected frames in an array

frames = []

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)   

#%%
# Display median frame

cv2.imshow('frame', medianFrame)

cv2.waitKey(0)

#%%
cv2.imwrite('test2.png', medianFrame)


