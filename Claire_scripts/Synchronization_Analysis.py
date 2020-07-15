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
from auto_filter_full_CE import  auto_scoring_tracefilter_full, transform_data
from find_features_CE import features
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.video.compositing as mp
import gizeh
import os
from glob import glob
from functions_test import binarize_Op_2, manual_scoring
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

# Filtering data and extracting features

filtered_df=new_features.filter_df(data_auto1)

new_features.fit(filtered_df,filter_feature=True,fill_na=True,estimate_na=False)

#%%

# Looping the operculum feature based on length of the animation display
LoopLength = 1080
Variable = new_features.operculum
NumLoops = int(int(len(Variable))/LoopLength)
Looped = pd.DataFrame(np.reshape(Variable.values[:(LoopLength*NumLoops)],(NumLoops, LoopLength))).T

#%%
Rates = []
# binarize and calculate rate for each column

for i in np.arange(len(Looped.columns)):
    binarized_col = binarize_Op_2(Looped[Looped.columns[i]], threshold = 70)
    bin_col_sum = (binarized_col == 1).sum()
    Rate = bin_col_sum/len(binarized_col)
    Rates.append(Rate)

# Rates should be a length of 22, plot to see if it goes up down up down

#%%
plt.plot(Rates)

#%%
#Decoding, make rates into a dataframe,  paste ListA and make that another column
# either export or copy paste into an excel

Rates_df = pd.DataFrame(Rates)
ListA = [3,4,2,4,3,4,3,4,2,4,2,4,3,4,2,4,2,4,3,4,2,4]
Rates_df['color'] = ListA

White_auto = Rates_df[Rates_df['color'] == 4].mean()[0]
Blue_auto = Rates_df[Rates_df['color'] == 2].mean()[0]
Red_auto = Rates_df[Rates_df['color'] == 3].mean()[0]

print('White', White_auto, 'Blue', Blue_auto, 'Red' , Red_auto)

Rates_df.to_excel("907.xlsx")

#%%

index = (Blue_auto - Red_auto) /(Blue_auto + Red_auto)
print(index)

#%%

## MANUAL SCORING

## Load Data
excel_files = glob(os.path.join('.', '*.xlsx'))
#    
file_handle1 = excel_files[-1]
data_manual1 = pd.read_excel(file_handle1)

Manual1 = manual_scoring(data_manual1, Operangle)

#%%
# MANUAL SCORING

Looped_Manual = pd.DataFrame(np.reshape(Manual1.values[:(LoopLength*NumLoops)],(NumLoops, LoopLength))).T

Rates_manual = []
# binarize and calculate rate for each column

for i in np.arange(len(Looped_Manual.columns)):
    bin_col_sum = Looped_Manual[Looped_Manual.columns[i]].sum()
    Rate = bin_col_sum/len(Looped_Manual[Looped_Manual.columns[i]])
    Rates_manual.append(Rate)

#%%

## MANUAL SCORING

plt.plot(Rates_manual)
#%%
#Decoding, make rates into a dataframe,  paste ListA and make that another column
# either export or copy paste into an excel

Rates_df_manual = pd.DataFrame(Rates_manual)
ListA = [3,4,2,4,3,4,3,4,2,4,2,4,3,4,2,4,2,4,3,4,2,4]
Rates_df_manual['color'] = ListA

White_manual = Rates_df_manual[Rates_df_manual['color'] == 4].mean()[0]
Blue_manual = Rates_df_manual[Rates_df_manual['color'] == 2].mean()[0]
Red_manual = Rates_df_manual[Rates_df_manual['color'] == 3].mean()[0]

print('White', White_manual, 'Blue', Blue_manual, 'Red' , Red_manual)

Rates_df_manual.to_excel("894_Manual_Results.xlsx")

#%%

## MANUAL SCORING

index_manual = (Blue_manual - Red_manual) /(Blue_manual + Red_manual)
print(index_manual)

#%%
# Measuring synchronization
indbins = np.random.rand(0,LoopLength)
#
for i in np.arange(0,len(Looped.columns)):
    indbin = uniform(binarizeOp(Looped_Manual[i], threshold = 0), LoopLength)
    indbins = np.vstack([indbins, indbin])



