#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:19:00 2020

@author: ryan
"""

import pickle
from tqdm import tqdm
import pandas as pd
from find_features import features
from contour_utils import *
import numpy as np

with open("data/head_indexes", "rb") as fp:  
    head_indexs = pickle.load(fp)

with open("data/tail_indexes", "rb") as fp:   
    tail_indexs = pickle.load(fp)
with open("data/fish_segment_length", "rb") as fp:  
    lengths = pickle.load(fp)
with open("data/curviness_score", "rb") as fp:   
    curve_scores = pickle.load(fp)  
with open("data/contour_array", "rb") as fp:  
    new_contour_array = pickle.load(fp)  


path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
df=df.iloc[90000:95000,:]
new_features=features(starttime=0,endtime=5000)
filtered_df=new_features.filter_df(df,add_midpoint=True)
#filtered_df=inside_mask(df,mask_array,kernel_size=11)
#filtered_df=filtered_df.fillna(method="ffill")
filtered_head=relative_position_check(filtered_df.A_head)
filtered_head=filtered_head.fillna(method="ffill")
    
filtered_df=filtered_df.fillna(method="ffill")
spine1=filtered_df.F_spine1

#head=filtered_df.A_head
head_x=filtered_head.x
head_y=filtered_head.y

head_midlines=[]
for i in tqdm(range(len(new_contour_array))):
    contour=new_contour_array[i]
    contour=contour.squeeze()
    N=len(contour)
    head_index=better_head_indexs[i]
    midline=np.zeros((0,2),dtype=np.float64)
    for j in [0,150]:
        midline=np.concatenate((midline,((contour[(head_index+j)%N]+contour[(head_index-j)%N])/2).reshape(1,2)),0)
    head_midlines.append(midline) 
    
tail_midlines=[]
for i in tqdm(range(len(new_contour_array))):
    contour=new_contour_array[i]
    contour=contour.squeeze()
    N=len(contour)
    tail_index=tail_indexs[i]
    midline=np.zeros((0,2),dtype=np.float64)
    for j in [0,150]:
        midline=np.concatenate((midline,((contour[(tail_index+j)%N]+contour[(tail_index-j)%N])/2).reshape(1,2)),0)
    tail_midlines.append(midline) 

tail_angle=[]
for i in tqdm(range(len(new_contour_array))):
    head_midline=head_midlines[i]
    tail_midline=tail_midlines[i]
    vec1=np.array(head_midline[0]-head_midline[1])
    vec2=np.array(tail_midline[0]-tail_midline[1])
    cos=np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    angel=np.arccos(cos)/np.pi*180
    tail_angle.append(angel)

distance_to_spine=[]
intersections=[]
for i in range(len(new_contour_array)):
    head_midline=head_midlines[i]
    tail_midline=tail_midlines[i]
    headx1,heady1=head_midline[0]
    headx2,heady2=spine1.iloc[i,:2]
    #headx2,heady2=head_midline[1]
    tailx,taily=tail_midline[0]
    if headx1==headx2:
        d=abs(tailx-headx1)
        k1=10000
        b1=-k1*headx1
    elif heady1==heady2:
        d=abs(taily-heady1)
        k1=0
        b1=heady1
    else:
        k1=(heady1-heady2)/(headx1-headx2)
        b1=heady2-k1*headx2
        d=abs(k1*tailx-taily+b1)/np.sqrt(k1**2+1)
    try:
        k2=-1/k1
        b2=taily-tailx*k2
    except ZeroDivisionError:
        k2=10000
        b2=-k2*tailx      
    try:
        x=(b2-b1)/(k1-k2)
        intersect=(x,k1*x+b1)
        intersections.append(intersect)
    except ZeroDivisionError:
        #use midpoint of head and tail, assume it's just the same line
        intersections.append(((headx1+tailx)/2,(heady1+taily)/2))
    
    
def plot_result(curve_score,tail_index,head_index,intersect,time,img_size=600,to_array=False,vmax=1):
    contour=curve_score[:,:2]
    fig=plt.figure()      
    plt.scatter(curve_score[:,0], curve_score[:,1], c=curve_score[:,2], s=3,cmap="YlGnBu", vmin=0, vmax=vmax)
    plt.plot(np.float64(contour[tail_index,0]),np.float64(contour[tail_index,1]),"ro",markersize=5,color="red")
    plt.plot(np.float64(contour[head_index,0]),np.float64(contour[head_index,1]),"ro",markersize=5,color="purple")
    head=contour[head_index]
    tail=contour[tail_index]
    pt1=intersect+1000/np.linalg.norm(np.array([head[0]-intersect[0],head[1]-intersect[1]]))*np.array([head[0]-intersect[0],head[1]-intersect[1]])
    pt2=intersect+1000/np.linalg.norm(np.array([tail[0]-intersect[0],tail[1]-intersect[1]]))*np.array([tail[0]-intersect[0],tail[1]-intersect[1]])   
    plt.plot([intersect[0],head[0]],[intersect[1],head[1]],linewidth=1, linestyle = "-", color="red")
    plt.plot([intersect[0],tail[0]],[intersect[1],tail[1]],linewidth=1, linestyle = "-", color="red")
    plt.plot(intersect[0],intersect[1],"ro",markersize=5,color="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.text(-300,80,"{} frame".format(time),bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    if not to_array:
        plt.show()
    else:
        out=mplfig_to_npimage(fig)
        plt.close(fig)
        return out
def make_frame(t):
    time=int(t*40)
    curve_score=curve_scores[time]
    tail_index=tail_indexs[time]
    head_index=better_head_indexs[time]
    intersect=intersections[time]
    return plot_result(curve_score,tail_index,head_index,intersect=intersect,time=time,img_size=1500,to_array=True,vmax=1)

animation = VideoClip(make_frame, duration = 125)
animation.write_videofile("videos/dist_to_spine4.mp4", fps=40)
  
    
    
    
    

