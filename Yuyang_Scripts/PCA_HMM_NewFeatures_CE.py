#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:38:55 2020

@author: Claire
"""

# load packages

import numpy as np
import pandas as pd


from auto_filter_full import  auto_scoring_tracefilter_full, transform_data, get_data

from find_features import features


import ssm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import interpolate
import seaborn as sns; sns.set()
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook as tqdm
from sklearn.cluster import KMeans
from itertools import permutations
import moviepy.video.compositing as mp
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
import gizeh


#%%
def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    
    return (data['x'],data['y'])

def speed (data, fps = 40):
    
    ''' function that calculates velocity of x/y coordinates
    plug in the xcords, ycords, relevant dataframe, fps
    return the velocity as column in relevant dataframe'''
    poi = ['A_head']
    (Xcoords, Ycoords)= coords(data[poi[0]])
    distx = Xcoords.diff() 
    disty = Ycoords.diff()
    TotalDist = np.sqrt(distx**2 + disty**2)
    Speed = TotalDist / (1/fps) #converts to seconds
#    
    return Speed

def midpoint (pos_1, pos_2, pos_3, pos_4):
    '''
    give definition pos_1: x-value object 1, pos_2: y-value object 1, pos_3: x-value object 2
    pos_4: y-value object 2
    '''
    midpointx = (pos_1 + pos_3)/2
    midpointy = (pos_2 + pos_4)/2

    return (midpointx, midpointy)
    

#%%
#load the data

home_dir = '.'
h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
print(h5_files)

data_auto1 = get_data(h5_files)
#%%
df=transform_data(data_auto1)
#%%
focus_df=df[['A_head',"F_spine1",'mid_spine1_spine2',"G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6",
             "L_spine7","B_rightoperculum",
                 'E_leftoperculum']]

filtered_df=auto_scoring_tracefilter_full(focus_df)

#%%
# define new features
starttime = 90000
duration = 200
new_features=features(starttime=starttime, duration = duration)
new_features.fit(filtered_df)

#%%

# make a DataFrame

Test = pd.DataFrame()
#operculum
Test['operculum'] = np.array(new_features.operculum)
#orientation
Test['orientation'] = np.array(new_features.ori)
#curve_diff
Test['curve_diff'] = np.array(new_features.diff_curvature[0])
#Velocity
Test['Velocity'] = np.array(speed(filtered_df)[starttime:starttime+(duration*40)])
#Xlocation
Test['X_head'] = np.array(filtered_df['A_head']['x'][starttime:starttime+(duration*40)])


Test = Test.fillna(method = 'bfill')
Test = Test.fillna(method = 'ffill')
#%%

# #Test could also look like this, with all the curv diffs
Test2 = pd.DataFrame(new_features.diff_curvature)
Test2 = Test['operculum']
Test2 = Test['orientation']
Test2 = Test.fillna(method = 'bfill')
Test2 = Test.fillna(method = 'ffill')

#%%
PCA_data = Test2.iloc
data_scaled = StandardScaler().fit_transform(Test2)
pca = PCA()
pca.fit(data_scaled)
pcs = pca.transform(data_scaled)

pcs = pcs[:,:2]
pcs = np.clip(pcs, -3, 3)
plt.scatter(pcs[:,0], pcs[:,1], s = 1)

#%%
# Now incorporate the manual scoring of Elaine
#Manual Scoring
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

## Manual Scoring
excel_files = glob(os.path.join('.', '*.xlsx'))
#    
file_handle1 = excel_files[-1]
data_manual1 = pd.read_excel(file_handle1)

#%%
Manual1 = manual_scoring(data_manual1, data_auto1, crop0 = starttime, crop1 =  starttime+(duration*40))


#%%
Test['Manual'] = np.array(Manual1)
data_scaled = StandardScaler().fit_transform(Test2)
pca = PCA()
pca.fit(data_scaled)
pcs = pca.transform(data_scaled)


pcs = pcs[:,:2]
pcs = np.clip(pcs, -3, 3)
plt.scatter(pcs[:,0], pcs[:,1], s = 1, c = Test['Manual'],cmap = 'cividis' )
plt.legend(Test['Manual'])

#%%

#kmeans applied directly to operculum x orientation
kmeans=KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=10)
kmeans.fit(np.array([new_features.operculum,new_features.ori]).T)
labels=kmeans.predict(np.array([new_features.operculum,new_features.ori]).T)
plt.figure(dpi=300)
plt.scatter(x=new_features.operculum,y=new_features.ori,s=1, c=labels, cmap='cividis')


#%%

# using HMM to define clusters
hmm_3=hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
hmm_3.fit(pcs)
y=hmm_3.predict(pcs)
plt.scatter(pcs[:, 0], pcs[:, 1], c=y,s=1, cmap='viridis')

#%%


#%%
'''
The next step is to create an HMM based on the curve_diff to see if defines states
'''
xx = Test2['Velocity']
yy = Test['X_head']
# zz = Test['curve_diff']# Needs work, but will help separate out the resting from the tail beating eventually
data = pd.DataFrame(np.column_stack((xx,yy)))


## feature exploration
plt.hist(xx) # right skewed, log transformation?
plt.hist(yy)

plt.boxplot

#%%
## setting hmm
# Set the parameters of the HMM
T = 8000   # number of time bins
K = 4      # number of discrete states
D = 2      # data dimension

## use EM to infer the model
data_em = np.array(data)
N_iters = 50

hmm = ssm.HMM(K, D, observations="gaussian")
hmm_lls = hmm.fit(data_em, method="em", num_em_iters=N_iters)
inferred_states = hmm.most_likely_states(data_em)

## plot EM results
plt.plot(hmm_lls, label="EM")
# plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")

## plot the inferred states
plt.figure(figsize=(8, 4))
plt.imshow(inferred_states[None, :], aspect="auto",cmap = "gray")
plt.xlim(0, T)
plt.ylabel("inferred\nstate")
plt.yticks([])
plt.title("states plot using peri and turning angle and speed")
plt.show()

## transition matrix
learned_transition_mat = hmm.transitions.transition_matrix

fig = plt.figure(figsize=(8, 4))

im = plt.imshow(learned_transition_mat, cmap='gray')
plt.title("Learned Transition Matrix using peri and orientation")

cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

#%%

#Visualizing point in video when state is being shown

# make and save a clip that shifts in color depending on the value of the inferred_states
duration =200
fps = 40

def make_frame(time):
    timeint = int(time*fps)
    col1 = y[timeint]
    col1 = col1/4
    print(col1)
    surface = gizeh.Surface(128,128, bg_color=(1,1,1,.5))
    if col1 == 0:
        circle = gizeh.circle(50, xy = (64,64), fill=(1,1,1))
    if col1 == .25:
        circle = gizeh.circle(50, xy = (64,64), fill=(0,0, 0))
    if col1 == .5:
        circle = gizeh.circle(50, xy = (64,64), fill=(.5,0,0))
    if col1 == .75:
        circle = gizeh.circle(50, xy = (64,64), fill=(0,0,.75))
    circle.draw(surface)
    return surface.get_npimage()
    
animation = VideoClip(make_frame, duration = duration)
animation.write_videofile("HMM_code.mp4", fps=40)

#%%
# compose the original video (of the proper duration and time in video with the color coded video)
clip1 = VideoFileClip("IM1_IM22.1.1.mp4").subclip(starttime/40,starttime/40+duration)
clip2 = VideoFileClip("HMM_code.mp4")
final_clip = mp.CompositeVideoClip.CompositeVideoClip([clip1,clip2])
final_clip.write_videofile("Vel_curv_diff.mp4", fps = 40)

#%%
#Make a dendrogram of the different states that exist within an aggressive state, this aggressive state is defined as the closeness
# of hte animal to the screen and the velocity of the fish, and wiggle of the tail (is split into only 2 states so will avoid false positives)
# Then within these aggressive boughts, I am looking for how many behavioral clusters exist wihtin this period of time. Build a dendrogram of 
# only the "on" times to see how many behaviors exist and then understand the transition probabilities between them.

# take the kmeans and make an HMM