#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:42:08 2020

@author: ryan
"""

from pipeline_functions import *
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.ndimage import zoom  
from contour_utils import *
from find_features import features
from scipy import sparse
from LoopingArray import LoopingArray
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
t1=datetime.now()
IMAGELENGTH=500
fps=40
videopath="TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"
#conservative_mask

path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
df=df.iloc[70000:150000,:]
new_features=features(starttime=0,endtime=80000)
filtered_df=new_features.filter_df(df,add_midpoint=True)
new_features.fit(filtered_df)
other_features,_,_,_=new_features.export_df()
filtered_head=filtered_df.A_head.interpolate(method="nearest")
filtered_head=relative_position_check(filtered_head,max_dist=60,max_counter=30)
filtered_head=filtered_head.interpolate(method="nearest")

contour_array=find_conservative_mask(videopath,length=80000,start=70000,step=1)

with open("data/IM1_IM2_2.1.1_L_70000_150000_contour_raw_NOV12", "wb") as fp:
    pickle.dump(contour_array, fp)

#more accurate contour and other features

new_contour_array=find_tail(videopath,contour_array,filtered_head,start=70000,step=1,interpolate=False,size=IMAGELENGTH)

with open("data/IM1_IM2_2.1.1_L_70000_150000_contour_refined_NOV12", "wb") as fp:
    pickle.dump(new_contour_array, fp)

curve_scores=[]
tail_indexs=[]
better_head_indexs=[]
lengths=[]
head_x=np.array(filtered_head.x)
head_y=np.array(filtered_head.y)
for i in tqdm(range(len(new_contour_array))):
    contour=new_contour_array[i]
    #since image is interpolated, and its size is 3 times before
    head_index=head_on_contour(head_x[i], head_y[i], contour)
    better_head_index,tail_index,curve_score,length=predict_tail(contour,head_index,step=[33,57],neighbor_width=[17,17],
                                                                 frame=i)
    tail_indexs.append(tail_index)
    curve_scores.append(curve_score)
    lengths.append(length)
    better_head_indexs.append(better_head_index)

head_indexs=pd.Series(better_head_indexs).fillna(method="ffill").astype(int)
tail_indexs=pd.Series(tail_indexs).fillna(method="ffill").astype(int)


with open("data/IM1_IM2_2.1.1_L_70000_150000_curve_scores_NOV12", "wb") as fp:
    pickle.dump(curve_scores, fp)
    
with open("data/IM1_IM2_2.1.1_L_70000_150000_tail_index_NOV12", "wb") as fp:
    pickle.dump(tail_indexs, fp)   
    
with open("data/IM1_IM2_2.1.1_L_70000_150000_head_index_NOV12", "wb") as fp:
    pickle.dump(better_head_indexs, fp)    

with open("data/IM1_IM2_2.1.1_L_70000_150000_fish_segment_length_NOV12", "wb") as fp:
    pickle.dump(lengths, fp)     

tail_angles=[]
tail_devs=[]
for i in range(len(new_contour_array)):
    contour=curve_scores[i][:,:2]
    contour=contour.squeeze()
    N=len(contour)
    head_index=head_indexs[i]
    tail_index=tail_indexs[i]
    tail_angle,tail_dev=compute_TailAngle_Dev(head_index,tail_index,contour)
    tail_angles.append(tail_angle)
    tail_devs.append(tail_dev)
tail_angles=np.array(tail_angles)
tail_devs=np.array(tail_devs)

other_features["Tail_Angle"]=tail_angles
other_features["Tail_Dev"]=tail_devs
other_features['X_Position']=filtered_head.x
other_features.to_csv("data/IM2_IM2_2.1.1_L_data_auto_scored.csv")
t2=datetime.now()
print("time_elapsed={}".format(t2-t1))