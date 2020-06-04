#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:16:25 2020

@author: richardpham
"""


#%%

# you'll need to pip install moviepy

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import interpolate
import seaborn as sns; sns.set()
from hmmlearn import hmm
# import ssm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook as tqdm
from sklearn.cluster import KMeans
from itertools import permutations
import moviepy
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
#%%

# Prep functions: loads data, performs basic arithmetic

def auto_scoring_tracefilter(data,p0=20,p2=15):
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum', 'C_tailbase', 'D_tailtip','E_leftoperculum']
    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
            mydata[b][j][xdiff_check] = np.nan
 
            origin_check = mydata[b][j] < p2
            mydata[origin_check] = np.nan

    return mydata



def getfiltereddata_2(h5_files):
    file_handle1 = h5_files[0]

    with pd.HDFStore(file_handle1,'r') as help1:
        data_auto1 = help1.get('df_with_missing')
        data_auto1.columns= data_auto1.columns.droplevel()
        data_auto1_filt = auto_scoring_tracefilter (data_auto1)
     
    file_handle2 = h5_files[1]

    with pd.HDFStore(file_handle2,'r') as help2:
        data_auto2 = help2.get('df_with_missing')
        data_auto2.columns= data_auto2.columns.droplevel()
        data_auto2_filt = auto_scoring_tracefilter(data_auto2)
        data_auto2_filt['A_head']['x'] = data_auto2_filt['A_head']['x'] + 500
        data_auto2_filt['B_rightoperculum']['x'] = data_auto2_filt['B_rightoperculum']['x'] + 500
        data_auto2_filt['E_leftoperculum']['x'] = data_auto2_filt['E_leftoperculum']['x'] + 500
    
    
    data_auto1_filt['zeroed','x'] = data_auto1_filt['A_head']['x'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[0]
    data_auto1_filt['zeroed','y'] = data_auto1_filt['A_head']['y'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[1]
    
    data_auto2_filt['zeroed','x'] = data_auto2_filt['A_head']['x'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[0]
    data_auto2_filt['zeroed','y'] = data_auto2_filt['A_head']['y'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[1]
    
    
    return data_auto1_filt, data_auto2_filt


def mydistance(pos_1,pos_2):
    '''
    Takes two position tuples in the form of (x,y) coordinates and returns the distance between two points
    '''
    x0,y0 = pos_1
    x1,y1 = pos_2
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)

    return dist

    
def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    
    return (data['x'],data['y'])



# Measuring Tail Beating Behavior
def conditional_orientation(data, orientation, threshold):
    '''
    creates a dataframe of only frames in which the fish
    is turned inside a certain orientation.
    '''
    boolean = orientation.apply(lambda x: 1 if x > threshold else 0) 
    df = pd.DataFrame(data, index = boolean)
    df = df.loc[1]
    
    return df

#%%
# FILTER FUNCTION
    
# My attempt at a filter function. Still needs Yuqi's radius function and Yuyang's stepwise filter. Instead
# of permutation, should be stepwise like Yuyang's or Yuqi's
def Combine_filter_CE (data, p0 = 20  , p1 = 5):
    '''
    
    This function filters  based on 1) jumps between frames, 2) the liklihood of the position, 3) the closeness of the points
    It does not filter based on any distance (Yuqi and Yuyang can perfect)
    
    returns a whole dataframe with NANs according to general and specific spine criterion
    '''
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum','E_leftoperculum',"F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7", 'D_tailtip','C_tailbase']
    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
            mydata[b][j][xdiff_check] = np.nan
        threshold = mydata[b]['likelihood'].quantile(.5)
        lik_check = mydata[b]['likelihood'] > threshold
        mydata[b]['likelihood'][lik_check] = np.nan
    # spine_column=["A_head", "F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7",'D_tailtip','C_tailbase']
    # perm = permutations(spine_column, 2)
    # for i in list(perm):
    #     rel_dist = mydistance(coords(mydata[i[0]]),coords(mydata[i[1]]))
    #     print(mydata[i[0]])
    #     rel_dist_check = rel_dist < p1
    #     mydata[rel_dist_check] = np.nan 
    return(mydata)

#%%
    
#MOVIEPY FUNCTIONS BELOW

## This block of code makes a video of the points for a certain duration of time. Duration = 10 means 10 seconds
## of video. Fps is the frames per second. The # of frames you are using to make the video should be at least duration
## times 40. The name of the dataframe should be entered into the function, at least for now.
## make sure to change the name of the output video at least until everything is functionalized.
Focus_period = np.arange(8000, 9000)
Focus_data = data_auto1_filt
data = Focus_data[Focus_region] # define the data that you will be running through moviepy functions
duration = 10 
fps = 40
fig, ax = plt.subplots()

def make_frame(time):
    timeint = int(time*fps)
    ax.clear()
    x = data['A_head']['x'][timeint]
    y = data['A_head']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['F_spine1']['x'][timeint]
    y = data['F_spine1']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['G_spine2']['x'][timeint]
    y = data['G_spine2']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['H_spine3']['x'][timeint]
    y = data['H_spine3']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['I_spine4']['x'][timeint]
    y = data['I_spine4']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['I_spine4']['x'][timeint]
    y = data['I_spine4']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['J_spine5']['x'][timeint]
    y = data['J_spine5']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['K_spine6']['x'][timeint]
    y = data['K_spine6']['y'][timeint]
    ax.plot(x,y,'o')
    x = data['L_spine7']['x'][timeint]
    y = data['L_spine7']['y'][timeint]
    ax.plot(x,y,'o')
    
    ax.set_ylim([0,500])
    ax.set_xlim([0,500])
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)
animation.write_videofile("points.mp4", fps=40)

#%%

## Creates a list of pts arrays to prep for moviepy. 
## These functions will allow you to make a spline for any period of time
## for a dataframe. 
## you 
pts_all = []
x_all = []
y_all = []
def Combine_pts(data):
    for i in np.arange(len(data)):
        test_point = data.iloc[i,:]
        x = test_point[[0, 15, 18, 21, 24, 27, 30, 33]]
        y = test_point[[1, 16, 19, 22, 25, 28, 31, 34]]
        pts=np.vstack([x,y]).T
        pts=pts[~np.isnan(pts).any(axis=1)]
        pts_all.append(pts)
        print()
        if(pts.shape[0]>=4):
            tck,u=interpolate.splprep(pts.T, u=None, s=0.0)
            u_new = np.linspace(u.min(), u.max(), 100)
            x_new, y_new = interpolate.splev(u_new, tck, der=0)
            x_all.append(x_new)
            y_all.append(y_new)
    return( x_all, y_all)

x_all, y_all = Combine_pts(Combine_filter_CE(data))
duration = 10
fps = 40
fig, ax = plt.subplots()

def make_frame(time):
    timeint = int(time*fps)
    ax.clear()
    ax.plot(x_all[timeint], y_all[timeint], 'b--')
    ax.set_ylim([0,500])
    ax.set_xlim([0,500])
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)
animation.write_videofile("spline.mp4", fps=40)

#%%