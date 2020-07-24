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

#get the images,contours and masks
def filter_lowerright(image):
    #this is just a function to help ignore the sign "888" in the lower right corner
    image[500:,:]=255
    return image

##this is the contour
vidcap = cv2.VideoCapture("videos/Reverse_888_cut.mp4")
img_array=[]
mask_array=[]
contour_array=[]
for i in tqdm(range(40*300)):
    success,image=vidcap.read()
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
    img=cv2.GaussianBlur(img, (3, 3), 0) #still not sure if to gaussian blur smooth the image, I already add extra step length
    #when estimate the curvatures, and the blurred image is not as solid as I thought
    true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_array.append(true_contour[0])
    img_array.append(np.array(cv2.drawContours(np.float32(np.full(image.shape,255)),true_contour,0,(0,0,0),2),np.uint8))
    mask_array.append(np.array(cv2.drawContours(np.float32(np.full(image.shape,255)),true_contour,0,(0,0,0),cv2.FILLED),np.uint8))
    
 
    
#save result
out = cv2.VideoWriter('videos/reverse888_contour_blurred.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, (image.shape[0],image.shape[1]))
for i in range(len(img_array)):
     out.write(img_array[i])
out.release()

out = cv2.VideoWriter('videos/reverse888_mask_blurred.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, (image.shape[0],image.shape[1]))
for i in range(len(mask_array)):
     out.write(mask_array[i])
out.release()
#%%
#img_array is now point set of masks
#I try to filter the deeplabcut result based on whether they are inside the contour(dilated), but in the end I find 
#another way to calculate curvatue so it's not used at all
'''
path = "Yuyang_contour/Reverse_888DeepCut_resnet50_DLC_toptrackFeb27shuffle1_160000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
#the period I used in contour finding
df=df[12000:24000]


colors=["rosybrown","lightcoral","bisque","burlywood","darkorange","darkgoldenrod","gold","olive","olivedrab","forestgreen",
        "lightseagreen","darkslategrey","cyan","dodgerblue"]
labels=["head","Roper","tailbase","tailtip","Loper","spine1","spine2","spine3","spine4","spine5","spine6","spine7",
        "Leye","Reye"]
for i in range(10):
    img=cv2.cvtColor(img_array[i], cv2.COLOR_BGR2GRAY)
    eroded_img=cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (11,11)), iterations=1)
    plt.figure()
    plt.imshow(eroded_img,"gray")
    for j in range(14):
        plt.plot(df.iloc[i,j*3],df.iloc[i,j*3+1],"ro",markersize=2, color=colors[j],label=labels[j])
    plt.legend(loc="upper left",prop={'size': 10},bbox_to_anchor=(1.1, 1.05)) #should be better ways to set legends
    plt.show()

#a video about how original points looks like on contour
fig, ax = plt.subplots(dpi=150)
def make_frame(t):
    t=np.int(t*40)
    ax.clear()
    ax.imshow(img_array[t])
    for j in range(14):
        ax.plot(df.iloc[t,j*3],df.iloc[t,j*3+1],"ro",markersize=2, color=colors[j],label=labels[j])
    ax.legend(loc="upper left",prop={'size': 10},bbox_to_anchor=(1.1, 1.05)) #should be better ways to set legends
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = 300)

#filtering based on mask
def inside_mask(df,mask_array,kernel_size=11):
    data = df.copy()
    for i in tqdm(range(df.shape[0])):
        for j in range(int(df.shape[1]/3)):#how many features in the data
            x=int(np.round(df.iloc[i,j*3]))
            y=int(np.round(df.iloc[i,j*3+1]))
            mask=cv2.cvtColor(mask_array[i], cv2.COLOR_BGR2GRAY)
            k=cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
            eroded_mask=cv2.erode(mask, k, iterations=1)
            if eroded_mask[y,x]!=0:
                data.iloc[i,j*3]=np.nan
                data.iloc[i,j*3+1]=np.nan
    return data


filtered_df=inside_mask(df,mask_array)


#after_filtering comparision
def make_frame(t):
    t=np.int(t*40)
    ax.clear()
    ax.imshow(img_array[t])
    for j in range(14):
        ax.plot(filtered_df.iloc[t,j*3],filtered_df.iloc[t,j*3+1],"ro",markersize=2, color=colors[j],label=labels[j])
    ax.legend(loc="upper left",prop={'size': 10},bbox_to_anchor=(1.1, 1.05)) #should be better ways to set legends
    return mplfig_to_npimage(fig)

animation2 = VideoClip(make_frame, duration = 300)
final_clip = clips_array([[animation, animation2]]).subclip(0,60)
final_clip.write_videofile("videos/contour_bodyparts_compare.mp4",fps=40)

new_df=filtered_df.fillna(method="ffill").fillna(0)
'''

#%%
#calculate the curvature at each point of the contour,default step=3
def compute_pointness(contour, n=3):
    out=np.zeros((600,600))
    contour=contour.squeeze()
    N = len(contour)
    t=1/N
    for i in range(N):
        x_cur, y_cur = contour[i]
        x_next, y_next = contour[(i + n) % N]
        x_prev, y_prev = contour[(i - n) % N]
        dy1=(y_next-y_prev)/(2*n*t)
        dx1=(x_next-x_prev)/(2*n*t)
        dy2=(y_next+y_prev-2*y_cur)/(n*n*t*t)
        dx2=(x_next+x_prev-2*x_cur)/(n*n*t*t)
        curvature=(dx1*dy2-dy1*dx2)/(np.power(dx1*dx1+dy1*dy1,3/2))
        curvature=abs(curvature)
        out[x_cur,y_cur]=curvature
    return out


#test the curvature result, and it seems to make sense as the head or tail has the highest curvature because it's probably the sharpest,
#actions need to ignore that to get valid "curveness" in fish
contour=contour_array[0]
plt.figure()
for i in contour:
    plt.plot(i.flatten()[0],i.flatten()[1],"ro",markersize=1)

out=compute_pointness(contour, n=3)

def find_centroid(contour):
    moments=cv2.moments(contour)
    m01=moments.get("m01")
    m10=moments.get("m10")
    m00=moments.get("m00")
    xbar=m10/m00
    ybar=m01/m00
    return xbar,ybar

def compute_dist(contour,xbar,ybar):
    contour=contour.squeeze()
    dist=np.linalg.norm(contour-np.array([xbar,ybar]),axis=1)
    return dist



def plot_result(curvatures,contour):
    non_zero=np.nonzero(curvatures)
    x=non_zero[0]
    y=non_zero[1]
    xbar,ybar=find_centroid(contour)
    colors=np.array(np.zeros_like(x),np.float64)
    dists=compute_dist(contour,xbar,ybar)
    for i in range(len(x)):
        colors[i]=out[x[i],y[i]]
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,0.6),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.6)
    plt.plot(xbar,ybar,"ro",markersize=3,color="red")
    plt.xlim(0,600)
    plt.ylim(0,600)
    plt.colorbar()
    plt.title("curvature heatmap on fish contour")
    plt.show()

#test on first contour
contour=contour_array[0]
out=compute_pointness(contour)
plot_result(out,contour)

#test on randomly selected 10 contours
samples=np.random.randint(0,12000,10)

for i in samples:
    contour=contour_array[i]
    out=compute_pointness(contour,n=5)
    plot_result(out,contour)
    
test=contour_array[samples[1]]
find_centroid(test)

def filter_curvature(curvatures,contour):
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    quantile=np.quantile(dists,0.6)
    contour=contour.squeeze()
    N=len(contour)
    
    for i in range(N):
        x,y=contour[i]
        if dists[i]>quantile:
            curvatures[x,y]=0.1
    return curvatures
        
#test it again on the first contour
contour=contour_array[0]
out=compute_pointness(contour)
plot_result(out,contour)
filtered_output=filter_curvature(out,contour)
plot_result(filtered_output,contour)


#test on randomly selected 10 contours
samples=np.random.randint(0,12000,10)

for i in samples:
    contour=contour_array[i]
    out=compute_pointness(contour)
    filtered_output=filter_curvature(out,contour)
    plot_result(out,contour)


#compute the curvatures, maybe some tests on how valid the curvature later
#the bad thing is, it's likely the curvature is not accurately measures how fish curves its body, sometimes there might 
#just be a tweek in the tail or fin which creates a sharp area
#possible improvements:
#increase sample steps(n in compute pointness)
#gaussian blur?
#use cos angle instead
#filter out head and tail before hand?(i.e. it x+i/x-i reaches head/tail limit like quantile 0.9 or something, use that latest point)
  
from datetime import datetime
t1=datetime.now()
max_curvature=[]
for contour in tqdm(contour_array):
    curvatures=compute_pointness(contour)
    filtered_curvature=filter_curvature(curvatures,contour)
    max_curvature.append(np.max(filtered_curvature))
t2=datetime.now()
#1 min 17s for 12000 pts
print(t2-t1)

plt.plot(max_curvature)
plt.ylim(0,1)

def compute_cos(contour, n=3):
    out=np.zeros((600,600))
    contour=contour.squeeze()
    N = len(contour)
    t=1/N
    for i in range(N):
        x_cur, y_cur = contour[i]
        x_next, y_next = contour[(i + n) % N]
        x_prev, y_prev = contour[(i - n) % N]
        vec1=np.array([x_next-x_cur,y_next-y_cur])
        vec2=np.array([x_prev-x_cur,y_prev-y_cur])
        cos=np.sum(vec1*vec2)/np.sqrt(np.sum(vec1*vec1)*np.sum(vec2*vec2))
        out[x_cur,y_cur]=cos+1
    return out

#samples=np.array([2641,4080])
samples=np.random.randint(0,12000,2)

for i in samples:
    contour=contour_array[i]
    out=compute_cos(contour,n=9)
    plot_result(out,contour)

for i in samples:
    contour=contour_array[i]
    out=compute_pointness(contour,n=9)
    plot_result(out,contour)
    