#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:08:38 2020

@author: ryan
"""
'''
sorry my notebook's kernel keep dying because the video making process, so I can only use scripts for now

Basically I discard the features in spine7(too unstable) and collect all other features I defined previously to fit a HMM with
Gaussian distribution(using 4 or 8 states, should explore more options), plot the fitted spline and all the points with
inferred states and other features on it. Then stack it with the original video(VM1_VM2)

from the plot I can roughly see some patterns of the inferred states, like some states the fish is less aggressive and not
likely to face towards the glass, and when the fish makes a big turn it will always be predicted as one specific state(I 
guess it's due to the change of turning angle or the curvatures). But It is definitly worth more exploration by trying more
num_states and combination of features.

'''
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
from auto_filter_full import  auto_scoring_tracefilter_full, transform_data
path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
df=transform_data(df)
focus_df=df[['A_head',"F_spine1",'mid_spine1_spine2',"G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6",
             "L_spine7","B_rightoperculum",
                 'E_leftoperculum']]
filtered_df=auto_scoring_tracefilter_full(focus_df)
from find_features import features

new_features=features(starttime=100000,duration=300)
new_features.fit(filtered_df)

#drop spine7, it's too unstable
curvatures=new_features.curvatures.loc[:,:8]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler();scaler.fit(curvatures);curvatures=pd.DataFrame(scaler.transform(curvatures))

#use pca to lower the dimension of curvatures for each spline
from sklearn.decomposition import PCA
pca=PCA(n_components=5)
pcs=pd.DataFrame(pca.fit_transform(curvatures))
pcs.columns=["pc1","pc2","pc3","pc4","pc5"]
operculum=new_features.operculum
ori=new_features.ori
mov_speed=new_features.mov_speed
#turning angle is redefined by using spine1->spine1.5 to calculate the cosine.
turn_angle=new_features.turn_angle

starttime=100000
duration=281
fps=40
# the clusters are more evenly distributed when i scaled all the features, I can't explain the reason...but it's like that
operculum=np.array(operculum)
ori=np.array(ori)
mov_speed=np.array(mov_speed)
turn_angle=np.array(turn_angle)

scaler = StandardScaler();scaler.fit(operculum.reshape(-1,1));scaled_operculum=scaler.transform(operculum.reshape(-1,1)).reshape(12000,)
scaler = StandardScaler();scaler.fit(ori.reshape(-1,1));scaled_ori=scaler.transform(ori.reshape(-1,1)).reshape(12000,)
scaler = StandardScaler();scaler.fit(mov_speed.reshape(-1,1));scaled_mov_speed=scaler.transform(mov_speed.reshape(-1,1)).reshape(12000,)
scaler = StandardScaler();scaler.fit(turn_angle.reshape(-1,1));scaled_turn_angle=scaler.transform(turn_angle.reshape(-1,1)).reshape(12000,)

other_features=pd.DataFrame({"operculum":scaled_operculum,"orientation":scaled_ori,"movement_speed":scaled_mov_speed,
                                     "turning_angle":scaled_turn_angle},index=pd.RangeIndex(start=0, stop=12000, step=1))

all_features=pd.concat([pcs,other_features],axis=1)

model = hmm.GaussianHMM(n_components=8, covariance_type="full")
model.fit(all_features)
labels=model.predict(all_features)

#model = hmm.GaussianHMM(n_components=8, covariance_type="full")
#model.fit(other_features)
#labels=model.predict(other_features)
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#visualize the data with states and other features stated
#the kernel will die if I make a clip longer than 281 seconds... So I can only make the video this long
def make_frame_withgill(time):
    fig, ax = plt.subplots(dpi=300)
    t=np.int(starttime+time*fps)
    ax.clear()
    test_point = filtered_df.iloc[t,:]
    raw_test_point=focus_df.iloc[t,:]
    x = test_point[[0, 3, 6, 9, 12, 15,18,21,24,27]]
    y = test_point[[1, 4, 7, 10, 13, 16,19,22,25,28]]
    pts=np.vstack([x,y]).T
    ox = raw_test_point[[0, 3, 6, 9, 12, 15,18,21,24,27]]
    oy = raw_test_point[[1, 4, 7, 10, 13, 16,19,22,25,28]]
    raw_pts=np.vstack([ox,oy]).T
    if ~(np.isnan(pts[0,0]) or np.isnan(pts[0,1])):
        ax.text(raw_pts[0,0],raw_pts[0,1],"h",fontsize=12)
    if ~(np.isnan(test_point[30]) or np.isnan(test_point[31]) or np.isnan(test_point[33]) or np.isnan(test_point[34])):
        ax.scatter(test_point[[30,33]],test_point[[31,34]],s=10,color="purple")
        ax.plot(test_point[[30,0]],test_point[[31,1]],'b-',linewidth=2)
        ax.plot(test_point[[33,0]],test_point[[34,1]],'b-',linewidth=2)
    pts=pts[~np.isnan(pts).any(axis=1)]
    colormap = np.array(['b', 'brown', 'green','red','grey','pink','orange','black','gold','yellow'])
    categories = np.array(range(raw_pts.shape[0]))
    ax.scatter(raw_pts[:,0], raw_pts[:,1],s=10,c=colormap[categories])
    if(pts.shape[0]>=4):
        tck,u=interpolate.splprep(pts.T, u=None, s=0.0) 
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = interpolate.splev(u_new, tck, der=0)
        ax.plot(x_new, y_new, 'b-',linewidth=2)
    cur_label=labels[t-starttime]
    index=t-starttime
    labelmap = np.array(['b', 'silver', 'green','red','grey','pink','orange','black','gold','yellow'])
    ax.scatter(520,420,s=40,color=labelmap[cur_label])
    ax.text(520,480,"infered state is {}".format(cur_label),fontsize=6)
    ax.text(520,390,"pc1={}".format(all_features.iloc[index,0]),fontsize=6)
    ax.text(520,360,"pc2={}".format(all_features.iloc[index,1]),fontsize=6)
    ax.text(520,330,"pc3={}".format(all_features.iloc[index,2]),fontsize=6)
    ax.text(520,300,"pc4={}".format(all_features.iloc[index,3]),fontsize=6)
    ax.text(520,270,"pc5={}".format(all_features.iloc[index,4]),fontsize=6)
    ax.text(520,240,"operculum={}".format(operculum[index]),fontsize=6)
    ax.text(520,210,"orientation={}".format(ori[index]),fontsize=6)
    ax.text(520,180,"speed={}".format(mov_speed[index]),fontsize=6)
    ax.text(520,150,"turning angle={}".format(turn_angle[index]),fontsize=6)
    ax.grid(b=None)
    ax.set_ylim([0,500])
    ax.set_xlim([0,570])
    ax.patch.set_facecolor('white')
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame_withgill, duration = duration)
animation.write_videofile("IM1_IM2_L_inferred_8states.mp4", fps=40)

##stacking the original video with the newly generated video
import cv2
from moviepy.editor import *
from moviepy.video.fx.all import mirror_y
video = VideoFileClip("IM1_IM22.1.1DLC_resnet50_DLC_toptrackFeb27shuffle1_170000_labeled.mp4")
video=video.margin(10)
video=mirror_y(video)
video=video.subclip(2500,2781)
video_states = VideoFileClip("IM1_IM2_L_inferred_8states.mp4")
video_states=video_states.resize(0.5)
final_clip = clips_array([[video, video_states]])
final_clip.write_videofile("model_comparision_8states.mp4")
