#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:37:48 2020

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
from moviepy.video.io.bindings import mplfig_to_npimage
from tqdm import tqdm
from scipy.ndimage import zoom  
import warnings
from contour_utils import compute_cos,plot_result,find_centroid,visualize_steps,find_anchor_rle,visualize_segments,compute_dist,inside_mask
from scipy import interpolate 
from numpy import random
from find_features import features
import seaborn as sn
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import gizeh
import moviepy.video.compositing as mp
from scipy import sparse
#vidcap = cv2.VideoCapture("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4")
#success,image=vidcap.read()

image_length=500
fps=40
videopath="TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"
def find_conservative_mask(videopath,length=40*300,interpolate=True,start=0,step=1,pre_filter=None):
    '''
    prefilter:a function that changes mask array
    '''
    vidcap = cv2.VideoCapture(videopath)
    img_array=[]
    mask_array=[]
    contour_array=[]
    length=int(length/step)
    index=start
    vidcap.set(1,index)
    for i in tqdm(range(length)):
        success,image=vidcap.read()
        if success!=1:
            print("process stops early at {}th iteration".format(i))
            break
        index+=step
        vidcap.set(1,index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        #th=cv2.GaussianBlur(th, (3, 3), 0)
        #first shrink the mask to get it separate from its reflection
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        th = cv2.dilate(th, k2, iterations=1)
        #filter specific large black area
        if pre_filter is not None:
            th=pre_filter(th)
        contours, hierarchy = cv2.findContours(np.invert(th.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#simplified contour pts
        #finding the contour with largest area
        flag=0
        area=0
        for cnt in contours:
            new_area=cv2.contourArea(cnt)
            if new_area>area and new_area<10000 :#in case some contour contains almost the whole image, not required if invert the image first
                area=new_area
                fish_contour=cnt
                flag=1
        if flag==0:
            print("no valid fish contour find at index {}".format(i))
            img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
        else:
            #draw only the mask of this contour
            img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img=np.invert(np.array(img,np.uint8))
        #zoom the mask a little bit because we shrink it before
        k3=cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
        img=np.invert(np.array(cv2.erode(img,k3,iterations=1),np.uint8))
        #smoothing
        if interpolate==True:
            img=zoom(img,3)
            img=cv2.medianBlur(img, 21) #since zoom made the img larger
        else:
             img=cv2.medianBlur(img, 5)
        true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#complete contour pts
        try:
            contour_array.append(true_contour[0])
        except:
            contour_array.append(np.zeros((1,2),dtype=np.float64))
        img_array.append(np.array(cv2.drawContours(image.copy(),true_contour,0,(0,255,0),2),np.uint8))
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1]),255)),true_contour,0,(0,0,0),cv2.FILLED),np.uint8))
    return img_array,mask_array,contour_array

img_array,mask_array,contour_array=find_conservative_mask("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4",length=5000,start=90000,step=1,interpolate=False)
out = cv2.VideoWriter('videos/IM1_IM2_conservativemask_erode5_test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(img_array)):
     out.write(img_array[i])
out.release()



samples=random.randint(0,2000,5)
for sample in samples:
    plt.figure()
    plt.imshow(img_array[sample])

'''
plt.imshow(img_array[0])
plt.axhline(450)
plt.axhline(85)
plt.axvline(100)
plt.axvline(490)
'''

def find_tail(videopath,conservative_masks,conservative_contours,head,interpolate=True,start=0,step=1,pre_filter=None):
    vidcap = cv2.VideoCapture(videopath)
    img_array=[]
    mask_array=[]
    contour_array=[]
    index=start
    vidcap.set(1,index)
    length=len(conservative_masks)
    def exclusion(y,x):
        #flag1=np.logical_or(y>450,y<85)
        #flag2=np.logical_or(x<100,x>485)
        dists1=compute_dist(np.vstack([x,y]).T,xbar,ybar)
        #new found part should not be too close to the centroid(as it is tail)
        flag1=dists1<thres
        dists2=compute_dist(np.vstack([x,y]).T,head_x,head_y)
        #new found part shall not be too close to the head, otherwise it is likely be reflection
        '''
        what if the head DLC find is actually not accurate even after filtering?
        idea1: Considering left eye and right eye as well, it the filtering function filters out too many pts in a row,
        just let it deem next n head pts as nan instead and then resume the normal process, and in the mean time the
        we only check distance between mask and L/R eye.
        idea2:use the previous filtering method instead?
        '''
        flag2=dists2<thres
        flag=np.logical_or(flag1,flag2)
        x=x[~flag]
        y=y[~flag]
        data=np.ones(len(x))
        output=np.invert(np.uint8(sparse.coo_matrix((data,(y,x)),shape=(image_length,image_length)).toarray()))
        return output    
    for i in tqdm(range(length)):
        success,image=vidcap.read()
        if success!=1:
            print("process stops early at {}th iteration".format(i))
            break
        index+=step
        vidcap.set(1,index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        th = cv2.erode(th, k1, iterations=1)
        #filtering the mask derived by direct thresholding using the previous derived conservative mask and head position
        conservative_contour=conservative_contours[i]
        conservative_mask=conservative_masks[i]
        xbar,ybar=find_centroid(conservative_contour)
        dists=compute_dist(conservative_contour,xbar,ybar)
        thres=np.quantile(dists,0.5)
        y,x=np.nonzero(np.invert(th))#black area, i.e. fish body
        head_x=head.x.iloc[i]
        head_y=head.y.iloc[i]
        th=exclusion(y,x)           
        th=np.minimum(th,conservative_mask)
        if pre_filter is not None:
            th=pre_filter(th)
        contours, hierarchy = cv2.findContours(np.invert(th.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        flag=0
        area=0
        for cnt in contours:
            new_area=cv2.contourArea(cnt)
            if new_area>area and new_area<10000 :#in case some contour contains almost the whole image, not required if invert the image first
                area=new_area
                fish_contour=cnt
                flag=1
        if flag==0:
            print("no valid fish contour find at index {}".format(i))
            img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
        else:
            img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #to get rid of small holes in the mask
        k3=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        img=np.invert(np.array(cv2.erode(img,k3,iterations=1),np.uint8))
        if interpolate==True:
            img=zoom(img,3)
            img=cv2.medianBlur(img, 21) #since zoom made the img larger
        else:
             img=cv2.medianBlur(img, 5)
        true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#complete contour pts
        try:
            contour_array.append(true_contour[0])
        except:
            contour_array.append(np.zeros((1,2),dtype=np.float64))
        img_array.append(np.array(cv2.drawContours(image.copy(),true_contour,0,(0,255,0),2),np.uint8))
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1]),255)),true_contour,0,(0,0,0),cv2.FILLED),np.uint8))
    return img_array,mask_array,contour_array
   

path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
df=df.iloc[90000:95000,:]
filtered_df=inside_mask(df,mask_array,kernel_size=11)
filtered_df=filtered_df.fillna(method="bfill")
head_x=filtered_df.A_head.x
head_y=filtered_df.A_head.y     
head=filtered_df.A_head
new_img_array,new_mask_array,new_contour_array=find_tail("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4",mask_array,contour_array,head,start=90000,step=1,interpolate=False)
out = cv2.VideoWriter('videos/includeTail_test2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(new_img_array)):
     out.write(new_img_array[i])
out.release()


'''
for i in range(1560,1600):
    plt.figure()
    plt.imshow(new_img_array[i])
'''
#plt.plot(x[542],y[354],"ro",markersize=2)


#this video is to look at whether the filer on head is good
def make_frame(t):
    time=int(t*fps)
    mask=mask_array[time]
    x=head_x.iloc[time]
    y=head_y.iloc[time]
    fig=plt.figure()
    plt.imshow(mask,"gray")
    plt.plot(x,y,"ro",markersize=3)
    plt.xlim(0,500)
    plt.ylim(0,500)
    out=mplfig_to_npimage(fig)
    plt.close(fig)
    return out

animation = VideoClip(make_frame, duration = 50)
animation.write_videofile("videos/filled_head.mp4", fps=fps)





