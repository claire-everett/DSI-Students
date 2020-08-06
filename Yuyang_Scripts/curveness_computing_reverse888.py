#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:16:15 2020

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
from contour_utils import compute_cos,plot_result,find_centroid,visualize_steps,find_anchor_rle,visualize_segments
from scipy import interpolate 

#get the images,contours and masks
image_length=600
fps=40
def filter_lowerright(image):
    #this is just a function to help ignore the sign "888" in the lower right corner
    image[500:,:]=255
    return image
#get the contours
def find_contour(videopath,length=40*300,interpolate=True,step=1):
    vidcap = cv2.VideoCapture(videopath)
    img_array=[]
    mask_array=[]
    contour_array=[]
    length=int(length/step)
    index=0
    for i in tqdm(range(length)):
        success,image=vidcap.read()
        index+=step
        vidcap.set(1,index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        th = cv2.dilate(th, k2, iterations=1)
    #th=cv2.erode(th,None,iterations=1)
        th=filter_lowerright(th)
        contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        flag=0
        area=0
        for cnt in contours:
            new_area=cv2.contourArea(cnt)
            if new_area>area and new_area<10000 :#in case some contour contains almost the whole image, not required if invert the image first
                area=new_area
                fish_contour=cnt
                flag=1
        if flag==0:
            img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
        else:
            img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=np.invert(np.array(cv2.erode(img,k2,iterations=1),np.uint8))
        if interpolate==True:
            img=zoom(img,3)
            img=cv2.GaussianBlur(img, (9, 9), 0) #since zoom made the img larger
        else:
             img=cv2.GaussianBlur(img, (3, 3), 0)
        true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_array.append(true_contour[0])
        img_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1],3),255)),true_contour,0,(0,0,0),2),np.uint8))
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1],3),255)),true_contour,0,(0,0,0),cv2.FILLED),np.uint8))
    return img_array,mask_array,contour_array
    
img_array,mask_array,contour_array=find_contour("videos/Reverse_888_cut.mp4",length=40*30,step=1)
'''
out = cv2.VideoWriter('videos/reverse888_contour_blurred.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(img_array)):
     out.write(img_array[i])
out.release()

out = cv2.VideoWriter('videos/reverse888_mask_blurred.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(mask_array)):
     out.write(mask_array[i])
out.release()
'''

#compute the curvatures, maybe some tests on how valid the curvature later
#the bad thing is, it's likely the curvature is not accurately measures how fish curves its body, sometimes there might 
#just be a tweek in the tail or fin which creates a sharp area
#possible improvements:
#increase sample steps(n in compute pointness)
#gaussian blur?
#use cos angle instead
#filter out head and tail before hand?(i.e. it x+i/x-i reaches head/tail limit like quantile 0.9 or something, use that latest point)
#average the result with different step length? increase consistency
#edge???



#%%
#get those curveness scores
samples=np.random.randint(0,1200,10)
for i in samples:
    contour=contour_array[i]
    out,_1,_2=compute_cos(contour,step=30,img_size=1800,min_step=30)
    plot_result(out,contour,img_size=1800)

#calculate curveness
cosines=[]
topCos=[]
seg1s=[]
seg2s=[]
l=len(contour_array)
for contour in tqdm(contour_array):
    out,seg1,seg2=compute_cos(contour,step=30,img_size=1800,min_step=30)
    topCos.append(np.sort(out.flatten())[::-1][:60])
    cosines.append(out)
    seg1s.append(seg1)
    seg2s.append(seg2)
    #for monitor
    #per=int(i/l*100)
    #print("\r"+"["+">"*per+" "*(100-per)+"]",end="")
    

#make video
quantile=0.5
img_size=1800
def make_frame(time):
    fig=plt.figure(dpi=100)
    t=np.int(time*fps)
    cosine=cosines[t]
    head_tail_x,head_tail_y=np.where(cosine<0)
    new_cosine=cosine.copy()
    new_cosine[new_cosine<0]=0
    x,y=np.nonzero(new_cosine)
    xbar,ybar=find_centroid(contour_array[t])
    colors=np.array(np.zeros_like(x),np.float64)
    for i in range(len(x)):
        colors[i]=new_cosine[x[i],y[i]]
    plt.scatter(head_tail_x,head_tail_y,s=3,c="lemonchiffon")
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.2)#
    
    plt.plot(xbar,ybar,"ro",markersize=3,color="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.title("curveness heatmap on fish contour")
    #ax.axis("off")
    out=mplfig_to_npimage(fig)
    plt.close(fig)
    return out
    
animation = VideoClip(make_frame, duration = 30)
animation.write_videofile("videos/curveness_step30_minstep30.mp4", fps=40)


#out put top N cos/curvature
#for i in tqdm(range(len(cosines))):
    #topCos[i]=np.sort(cosines[i].flatten())[::-1][:30]
#pc1 already explains more than 95% variance, so it's already enough
curveness=np.array(topCos)
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pcs=pca.fit_transform(curveness)
plt.hist(pcs)

#most points with high pc value are those with large curveness score
outliers_ind=np.where(pcs>1)[0]
import random
for i in range(10):
    ind=random.choice(outliers_ind)
    plot_result(cosines[ind],contour_array[ind],img_size=1800)

 
#%%
#the midline thing
midlines=[]
In1=[]
Out1=[]
In2=[]
Out2=[]
warning_cnt=0
for i in tqdm(range(len(contour_array))):
    contour=contour_array[i]
    contour=contour.squeeze()
    N=len(contour)
    in1,out1,in2,out2,warning_cnt=find_anchor_rle(contour=contour,warning_cnt=warning_cnt,quantile=0.55)
    if in1 is not None:
    #the index is given in a counter clockwise way
        In1.append(in1)
        Out1.append(out1)
        In2.append(in2)
        Out2.append(out2)
        l1=(out1-in1)%N
        l2=(out2-in2)%N
        midline=np.zeros((min(l1,l2),2),dtype=np.float64)
        for j in range(min(l1,l2)):
            midline[j]=(contour[(in1+j)%N]+contour[(out2-j)%N])/2
        midlines.append(midline)
    else:
        midlines.append(np.full(2,2),np.nan)
 #this is actually not good, happens that there are small segments counted in    
def visualize_midlines(t):
    time=int(t*40)
    contour=contour_array[time]
    midline=midlines[time]
    contour=contour.squeeze()
    fig=plt.figure(dpi=100)
    x=contour[:,0]
    y=contour[:,1]
    plt.scatter(x,y,s=0.1,color="lemonchiffon")
    xmid=midline[:,0]
    ymid=midline[:,1]
    in1=In1[time]
    out1=Out1[time]
    in2=min(len(contour)-1,In2[time])
    out2=min(len(contour)-1,Out2[time])
    x1=np.array([contour[in1][0],contour[out1][0],contour[in2][0],contour[out2][0]]) 
    y1=np.array([contour[in1][1],contour[out1][1],contour[in2][1],contour[out2][1]]) 
    plt.scatter(x1,y1,s=3,c="red")
    plt.scatter(xmid,ymid,s=0.1,color="blue")
    plt.xlim(0,1800)
    plt.ylim(0,1800)
    out=mplfig_to_npimage(fig)
    plt.close(fig)
    return out
    
animation = VideoClip(visualize_midlines, duration = 30)
animation.write_videofile("videos/midlines_Wanchor.mp4", fps=40)

plot_result(cosines[78],contour_array[78],img_size=1800,quantile=0.6)
plot_result(cosines[82],contour_array[82],img_size=1800,quantile=0.6)


#TEST
#SHOULD NOT GET MUCH WARNINGS WHEN FINDING SEGMENTS
In1=[]
Out1=[]
In2=[]
Out2=[]
warning_cnt=0
for i in tqdm(range(len(contour_array))):
    contour=contour_array[i]
    contour=contour.squeeze()
    N=len(contour)
    in1,out1,in2,out2,warning_cnt=find_anchor_rle(contour=contour,warning_cnt=warning_cnt,quantile=0.55)
    #the index is given in a counter clockwise way
    In1.append(in1)
    Out1.append(out1)
    In2.append(in2)
    Out2.append(out2)

#This should be how to find curveness along the midline?
#fitting a B-spline sometimes give me a werid error, probably due to bad shape or somethind
curvatures=[]
for i in tqdm(range(len(midlines))): 
    midline=midlines[i]
    try:
        tck,u=interpolate.splprep([midline[:,0],midline[:,1]], u=None, s=0.0)
        dx1,dy1=interpolate.splev(u,tck,der=1)
        dx2,dy2=interpolate.splev(u,tck,der=2)
        k=(dx1*dy2-dy1*dx2)/np.power((np.square(dx1)+np.square(dy1)),3/2)
        curvatures.append(abs(k))
    except:
        curvatures.append(np.zeros((len(midline),),np.float64))

    
