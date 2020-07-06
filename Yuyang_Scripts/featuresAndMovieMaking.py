#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:08:11 2020

@author: ryan
"""

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
from find_features import features
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.video.compositing as mp
import gizeh
path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
starttime=90000
duration=200
#duration=1500
fps=40
new_features=features(starttime=starttime,duration=duration)
filtered_df=new_features.filter_df(df,add_midpoint=True)


new_features.fit(filtered_df,filter_feature=True,fill_na=True,estimate_na=False)

other_features,curvatures,diff_curvatures,tangent=new_features.export_df()
#%%
#Preprocessing
def manual_scoring(data_manual,data_auto,crop0 = 0,crop1= -1):
    '''
    A function that takes manually scored data and converts it to a binary array. 
    
    Parameters: 
    data_manual: manual scored data, read in from an excel file
    data_auto: automatically scored data, just used to establish how long the session is. 
    
    Returns: 
    pandas array: binary array of open/closed scoring
    '''
    Manual = pd.DataFrame(0, index=np.arange(len(data_auto)), columns = ['OpOpen'])
    reference = data_manual.index
    
    
    for i in reference:
        Manual[data_manual['Start'][i]:data_manual['Stop'][i]] = 1
    
    print(Manual[data_manual['Start'][i]:data_manual['Stop'][i]]) 
     
    return Manual['OpOpen'][crop0:crop1]

excel_files = "TailManual.xlsx"
file_handle1 = excel_files
data_manual1 = pd.read_excel(file_handle1)
Manual1 = manual_scoring(data_manual1, df, crop0 = starttime, crop1 =  starttime+(duration*40))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

curvature_scaled = StandardScaler().fit_transform(curvatures.iloc[:,5:9])
pca_curvature = PCA()
pcs_curvature = pca_curvature.fit_transform(curvature_scaled)[:,:2]
pcs=pca_curvature.fit_transform(curvature_scaled)


diff_curvature_scaled = StandardScaler().fit_transform(diff_curvatures.iloc[:,:9])
pca_diff_curvature = PCA()
pcs_diff_curvature = pca_diff_curvature.fit_transform(curvature_scaled)[:,:5]

other_feature_scaled=StandardScaler().fit_transform(other_features)
pca_other = PCA()
pcs_other = pca_other.fit_transform(other_feature_scaled)

#Yuqi's code
loadings = pd.DataFrame(pca_curvature.components_.T,columns=['PC1', 'PC2',"3","4"])
loadings_abs = abs(loadings)
import seaborn as sn
sn.heatmap(loadings_abs, annot=True,cmap = "YlGnBu")

#%%
#fit HMM
#fig = plt.figure()
#ax1 = plt.axes(projection='3d')
#ax1.scatter3D(pcs_other[:, 0], pcs_other[:, 1],pcs_other[:, 2], s=1,c=Manual1,cmap='cividis')

#selected_diff_features=np.concatenate([pcs_diff_curvature,other_feature_scaled],axis=1)
plt.scatter(pcs_trunc[:,0],pcs_trunc[:,1])
selected_features=np.concatenate([pcs_curvature,other_feature_scaled],axis=1)
model = hmm.GaussianHMM(n_components=4, covariance_type="full")
#model.fit(selected_diff_features)

labels=model.predict(pcs_trunc)
#labels=model.predict(selected_diff_features)
plt.hist(labels)

#Try just use orientation
model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.fit(np.array(new_features.ori).reshape(-1,1))
labels=model.predict(np.array(new_features.ori).reshape(-1,1))
plt.hist(labels)
#transition matrix plot
learned_transition_mat = model.transmat_

fig = plt.figure(figsize=(8, 4))

im = plt.imshow(learned_transition_mat, cmap='gray')
plt.title("Learned Transition Matrix")

cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()
#%%
# make videos
def make_frame(time):
    timeint = int(time*fps)
    col1 = labels[timeint]
    col1 = col1/4
    #print(col1)
    surface = gizeh.Surface(128,128, bg_color=(1,1,1,.5))
    #fill= rgb/255
    if col1 == 0:
        circle = gizeh.circle(50, xy = (64,64), fill=(1,1,1))
    if col1 == .25:
        circle = gizeh.circle(50, xy = (64,64), fill=(0,0, 0))
    if col1 == .5:
        circle = gizeh.circle(50, xy = (64,64), fill=(.5,0,0))
    if col1 == .75:
        circle = gizeh.circle(50, xy = (64,64), fill=(0,0,.75))
    if col1 == 1.0:
        circle = gizeh.circle(50, xy = (64,64), fill=(1,0,1))
    circle.draw(surface)
    return surface.get_npimage()

manual=np.array(Manual1)
def make_frame_manual(time):
    timeint = int(time*fps)
    col1 = manual[timeint]
    col1 = col1/4
    #print(col1)
    surface = gizeh.Surface(128,128, bg_color=(1,1,1,.5))
    if col1 == 0:
        circle = gizeh.circle(50, xy = (64,64), fill=(1,1,1))
    if col1 == .25:
         circle = gizeh.circle(50, xy = (64,64), fill=(0,0,0))
    circle.draw(surface)
    return surface.get_npimage()

animation = VideoClip(make_frame, duration = 200)
animation.write_videofile("videos/HMM_4state_curv_2pc_code.mp4", fps=40)

animation_manual = VideoClip(make_frame_manual, duration = 200)
animation_manual.write_videofile("videos/Manual_tail_code.mp4", fps=40)

clip1 = VideoFileClip("videos/IM1_IM22.1.1DLC_resnet50_DLC_toptrackFeb27shuffle1_170000_labeled.mp4").subclip(starttime/40,starttime/40+duration)
clip2 = VideoFileClip("videos/HMM_4state_curv_2pc_code.mp4")
clip3=VideoFileClip("videos/Manual_tail_code.mp4")
final_clip = mp.CompositeVideoClip.CompositeVideoClip([clip1,clip2.set_position((0,0)),clip3.set_position((0,130))])
final_clip.write_videofile("videos/Vel_curv_2pcs.mp4", fps = 40)

'''
just pasting manual videos again because I used the wrong file...
clip= VideoFileClip("videos/Vel_diff_curv_partial.mp4")
final_clip=mp.CompositeVideoClip.CompositeVideoClip([clip,clip3.set_position((0,130))])
final_clip.write_videofile("videos/Vel_diff_curv_1200s.mp4", fps = 40)
'''