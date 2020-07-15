#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:04:27 2020

@author: ryan
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
from moviepy.video.fx.all import crop
from moviepy.editor import VideoFileClip, VideoClip, clips_array

#take 5min-10min as test period
from moviepy.editor import VideoFileClip    
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("videos/Reverse_888.mp4", 300, 600, targetname="videos/Reverse_888_cut.mp4")

bg = cv2.imread('test.png') # NN: reads image within backgroundImage folder
bgGray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
#%%
#structured way
#first dilate/erode the mask to seperate fish and its reflection,
#then find the contour with the largest area, as fish's contour
#erode/dilate the mask back to the original shape
#disadvantage:tailtip is often excluded




def filter_lowerright(image):
    #this is just a function to help ignore the sign "888" in the lower right corner
    image[500:,:]=255
    return image

vidcap = cv2.VideoCapture("videos/Reverse_888_cut.mp4")
img_array=[]
mask_array=[]
for i in range(40*300):
    success,image=vidcap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    th = cv2.dilate(th, k2, iterations=1)
    #th=cv2.erode(th,None,iterations=1)
    th=filter_lowerright(th)
    mask_array.append(th)
    contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    flag=0
    area=0
    for cnt in contours:
        new_area=cv2.contourArea(cnt)
        if new_area>area and new_area<10000 :#in case some contour contains almost the whole image
            area=new_area
            fish_contour=cnt
            flag=1
    if flag==0:
        img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
    else:
        img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=np.invert(np.array(cv2.erode(img,k2,iterations=1),np.uint8))
    true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_array.append(cv2.drawContours(image.copy(),true_contour,0,(0,255,0),3))
    
#make mp4
out = cv2.VideoWriter('videos/contour_dilate1_size7_test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, (bg.shape[0],bg.shape[1]))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

out = cv2.VideoWriter('videos/mask_dilate1_size7_test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, (bg.shape[0],bg.shape[1]),0)
for i in range(len(mask_array)):
    out.write(mask_array[i])
out.release()
#%%
#environment specific way
#manually exclude points if its x coordinate is lower or greater than some value
#disadvantages: 1.fickle, the reflection surface is not on the same vertical line, and looks like it depends on he fish's posture
#2.verbose...probably need to do this every time
# so it's a VERY BAD EXAMPLE BELOW


def filter_leftRight(image):
    image[:,:101]=255
    image[:,479:]=255
    return image

vidcap = cv2.VideoCapture("videos/Reverse_888_cut.mp4")
img_array=[]
mask_array=[]
for i in range(40*300):
    success,image=vidcap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    #th=cv2.erode(th,None,iterations=1)
    th=filter_leftRight(th)
    th=filter_lowerright(th)
    mask_array.append(th)
    contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    flag=0
    area=0
    for cnt in contours:
        new_area=cv2.contourArea(cnt)
        if new_area>area and new_area<10000 :#in case some contour contains almost the whole image
            area=new_area
            fish_contour=cnt
            flag=1
    if flag==0:
        img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
    else:
        img=cv2.drawContours(image.copy(),[fish_contour],0,(0,255,0),3)
    img_array.append(img)
    
out = cv2.VideoWriter('videos/contour_exclude_outsideTank_test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, (bg.shape[0],bg.shape[1]))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
