#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:03:01 2020

@author: ryan
"""
'''
The Code is very similar to Claire's Test_Run_Contour script except I tried to erode the thresholded mask rather than dilating it,
and use only the contour with the largest area
'''

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
from moviepy.video.fx.all import crop
from moviepy.editor import VideoFileClip, VideoClip, clips_array
vidcap = cv2.VideoCapture("videos/VM5_VM4_cropped_.mp4")
vidcap = cv2.VideoCapture("videos/IM1_IM22.1.1DLC_resnet50_DLC_toptrackFeb27shuffle1_170000_labeled.mp4")
vidcap = cv2.VideoCapture("videos/Reverse_888.mp4")
success,image=vidcap.read()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''
clip1 = VideoFileClip("videos/IM1_IM22.1.1DLC_resnet50_DLC_toptrackFeb27shuffle1_170000_labeled.mp4").subclip(0,60)
cropped_clip=crop(clip1,x1=90,x2=480,y1=90,y2=450)
cropped_clip.write_videofile("videos/test.mp4", fps = 40)
cropped_image=cropped_clip.to_ImageClip().get_frame(0)
image=clip1.to_ImageClip().get_frame(0)
'''
plt.imshow(image)
#plt.imshow(cropped_image)
#gray_blurred=cv2.GaussianBlur(gray, (21, 21), 0)  ## NN: Blurs image

#TO me use threshold 40 makes most sense
ret,th1 = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)
#th1 = cv2.dilate(th1, None, iterations=2)
plt.imshow(th1,"gray")

(contours, hierarchy) = cv2.findContours(th1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(image.copy(), contours, 15, (0,255,0), 3)
plt.imshow(img)


#trying to find a good threshold 40 works for me 
for i in range(30,50):
    plt.figure()
    ret,th1 = cv2.threshold(gray,i,255,cv2.THRESH_BINARY)
    plt.imshow(th1,cmap="gray")
    plt.title("threshold={}".format(i))


#Make a video for each frame, adding the coutourwith the largest area  
bg = cv2.imread('test.png') # NN: reads image within backgroundImage folder
bgGray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

vidcap = cv2.VideoCapture("videos/Reverse_888.mp4")
img_array=[]
for i in range(40*60):
    success,image=vidcap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fDelta=cv2.absdiff(bgGray, gray)
    ret,th = cv2.threshold(fDelta, 40, 255, cv2.THRESH_BINARY)
    th = cv2.erode(th, None, iterations=4)
    contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #finding the contour with the largest area
    fish_contour=contours[0]
    area=cv2.contourArea(fish_contour)
    for cnt in contours:
        new_area=cv2.contourArea(cnt)
        if new_area>area and new_area<10000 :#in case some contour contains almost the whole image
            area=new_area
            fish_contour=cnt
    img=cv2.drawContours(image.copy(),[fish_contour],0,(0,255,0),3)
    img_array.append(img)

#make mp4
out = cv2.VideoWriter('videos/contour_erode4.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, (bg.shape[0],bg.shape[1]))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


#%%
#below is my first attempt to find the contour directly instead of using cv2 findContour function
#So unless someone is very interested in the naive solution of finding contour, they can ignore this part
frames=[]
for i in range(2400):
    if success:
        success,image=vidcap.read()
        frames.append(image)

for i in range(1,11):
    gray=cv2.cvtColor(frames[i*40], cv2.COLOR_BGR2GRAY)
    plt.figure()
    ret,th1 = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)
    plt.imshow(th1,cmap="gray")
    plt.title("{}second".format(i))
    
def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image    

image=frames[2]
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im=np.ones((500,500,3))*255
ret,th1 = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)
ou=outline(im,th1/255,(0,255,0))
plt.imshow(ou)
plt.imshow(th1,"gray")
plt.imshow(image)

from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
clip1 = VideoFileClip("videos/VM5_VM4.mp4").subclip(0,60)

clip1.write_videofile("videos/VM5_VM4_cropped.mp4",fps=40)

clip2 = VideoFileClip("videos/VM5_VM4_cropped_.mp4")

for i in range(60):
    image=frames[i*40]
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,th1 = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
    ou=outline(image,th1/255,(0,255,0))
    plt.figure()
    plt.imshow(ou)
    plt.title("{}second".format(i))

fps=40
fig, ax = plt.subplots(dpi=100)
def make_frame(time):
        t=np.int(time*fps)
        ax.clear()
        image=frames[t]
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th1 = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
        ou=outline(image,th1/255,(0,255,0))
        ax.imshow(ou)
    #ax.axis("off")
        return mplfig_to_npimage(fig)
    
animation = VideoClip(make_frame, duration = 60)
animation.write_videofile("videos/thresholding_result_outline.mp4", fps=40)

fig, ax = plt.subplots(dpi=100)
def make_frame(time):
        t=np.int(time*fps)
        ax.clear()
        image=frames[t]
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th1 = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
        ax.imshow(th1)
    #ax.axis("off")
        return mplfig_to_npimage(fig)
    
animation = VideoClip(make_frame, duration = 60)
animation.write_videofile("videos/thresholding_result_outline.mp4", fps=40)


