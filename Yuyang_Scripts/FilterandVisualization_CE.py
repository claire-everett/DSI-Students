#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:05:11 2020

@author: Yuyang originally, turned into .py by Claire
"""

#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Conspecific_Tracking_ta_edit_117 as cs
from scipy import interpolate 
from scipy import misc
from beating import rotation
from beating import tail_spline
from functions import *
from hmmlearn import hmm

from auto_filter_full import  auto_scoring_tracefilter_full, transform_data, get_data
import moviepy.video.compositing as mp
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
from make_frame import make_frame_factory
from find_features import features

#%%

home_dir = '.'
h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
print(h5_files)

data_auto1 = get_data(h5_files)
#%%
df=transform_data(data_auto1)

#%%

# focus_cols=['A_head',"F_spine1",'mid_spine1_spine2',"G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6","L_spine7",'C_tailbase','D_tailtip']

## measuring Operculum without ANY filtering!?!?
operculum=auto_scoring_get_opdeg(df)
operculum=operculum.fillna(method="ffill")
plt.plot(operculum)

#%%


focus_df=df[['A_head',"F_spine1",'mid_spine1_spine2',"G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6",
             "L_spine7","B_rightoperculum",
                 'E_leftoperculum']]

filtered_df=auto_scoring_tracefilter_full(focus_df)

#%%
duration=60
fps=40
starttime=100000

#%%
curvatures=[]
count=0
baseline=np.array([filtered_df['mid_spine1_spine2']['x']-filtered_df['F_spine1']['x'],filtered_df['mid_spine1_spine2']['y']-filtered_df['F_spine1']['y']]).T
cos_derivative_baseline=[]
    #calculate the 3 features(curvature/diff_curvature/cos between the tangent line and baseline(spine1->spine1.5))
for i in range(starttime,starttime+duration*40):
    line=baseline[i,:]
    y=filtered_df.loc[i,[('A_head','y'),("F_spine1","y"),('mid_spine1_spine2',"y"),("G_spine2","y"),
                            ('mid_spine2_spine3',"y"),("H_spine3","y"),("I_spine4","y"),("J_spine5","y"),
                            ("K_spine6","y"),("L_spine7","y")]]
    x=filtered_df.loc[i,[('A_head','x'),("F_spine1","x"),('mid_spine1_spine2',"x"),("G_spine2","x"),
                            ('mid_spine2_spine3',"x"),("H_spine3","x"),("I_spine4","x"),("J_spine5","x"),
                            ("K_spine6","x"),("L_spine7","x")]]
    pts=np.vstack([x,y]).T
    index=~np.isnan(pts).any(axis=1)
    pts=pts[index]
    curvature=np.repeat(np.nan,10)
    cos=np.repeat(np.nan,10)
    if(pts.shape[0]>=4):
        tck,u=interpolate.splprep(pts.T, u=None, s=0.0)
        dx1,dy1=interpolate.splev(u,tck,der=1)
        dx2,dy2=interpolate.splev(u,tck,der=2)
        k=(dx1*dy2-dy1*dx2)/np.power((np.square(dx1)+np.square(dy1)),3/2)
        cos_=(dy1*line[1]+dx1*line[0])/np.linalg.norm(line)/np.sqrt(dy1*dy1+dx1*dx1)
        cos[index]=cos_
        curvature[index]=k
        curvatures.append(curvature)
        cos_derivative_baseline.append(cos)
    else:
        curvatures.append(curvature)
        cos_derivative_baseline.append(cos)
        count=count+1
#%%
count
#%%
#test=filtered_df.loc[157500:160000,:]
#%%
#np.sum(np.isnan(test))

#%%


duration=60
fps=40
starttime=100000
#%%
make_frame_withgill=make_frame_factory(starttime,duration,fps,focus_df,filtered_df)

#%%
#almost every single point in spine7 is filtered out
animation = VideoClip(make_frame_withgill, duration = duration)
animation.write_videofile("spline_IM1_IM2_R1e5.mp4", fps=40)

#%%
clip1 = VideoFileClip("IM1_IM22.1.1.mp4").subclip(50,100)
clip2 = VideoFileClip("spline_IM1_IM2_R1e5.mp4").subclip(50,100)
# final_clip = CompositeVideoClip([[clip1,clip2]])
# final_clip.write_videofile("result.mp4", fps = 40)
# final_clip = clips_array([[clip1, clip2]])
# final_clip.resize(width=480).write_videofile("my_stack.mp4")
final_clip = mp.CompositeVideoClip.CompositeVideoClip([clip2,clip1])
final_clip.write_videofile("result.mp4", fps = 40)
#%%

def overlayClips(clipTop, clipBottom):
    """
    Overlays a one clip on top of another
    Used for doing the gauss blur style bars on the video
    """
    return CompositeVideoClip([clipBottom.set_pos("center"), clipTop.set_pos("center")]) 

video = overlayClips(clip1, clip2)
#%%

new_features=features(starttime=100000)
new_features.fit(filtered_df)
#%%
new_features.visualize_cluster(num_cluster=2,s=1)
#cluster on 2 groups

#%%
#kmeans applied directly to operculum x orientation
kmeans=KMeans(n_clusters=2, init='k-means++', max_iter=1000, n_init=10)
kmeans.fit(np.array([new_features.operculum,new_features.ori]).T)
labels=kmeans.predict(np.array([new_features.operculum,new_features.ori]).T)
plt.figure(dpi=300)
plt.scatter(x=new_features.operculum,y=new_features.ori,s=1, c=labels, cmap='cividis')

#%%

