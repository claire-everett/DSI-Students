#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:48:09 2020

@author: Claire
"""
#%%
# import the packages
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate 
from scipy import misc
from hmmlearn import hmm
from auto_filter_full_CE import  auto_scoring_tracefilter_full_CE, transform_data
from find_features_CE import features
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.video.compositing as mp
import gizeh
import os
from glob import glob
#%%
# Load the data

home_dir = '.'#'/Users/Claire/Desktop/FishBehaviorAnalysis'
h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
file_handle1 = h5_files[0]

with pd.HDFStore(file_handle1,'r') as help1:
   data_auto1 = help1.get('df_with_missing')
   data_auto1.columns= data_auto1.columns.droplevel()

starttime = 0
duration = len(data_auto1)

new_features=features(starttime=starttime,duration=duration)

#%%
filtered_df=new_features.filter_df(data_auto1)

#%%
new_features.fit(filtered_df,filter_feature=True,fill_na=True,estimate_na=False)

#%%

