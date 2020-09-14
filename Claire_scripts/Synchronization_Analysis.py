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
from functions_test import binarize_Op_2 
import seaborn as sns; sns.set()
#%%
# Load the data

home_dir = '/Users/Claire/Desktop/PiColor/'#'/Users/Claire/Desktop/FishBehaviorAnalysis'
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
    binarized_col = binarize_Op_2(Looped[Looped.columns[i]], lb = 65, ub = 135)
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
from DSI_UsefulFunctions import manual_scoring
#%%
## Load Data
h5_dir = '/Users/Claire/Desktop/PiColor/ROC_Compare'
excel_files = sorted(glob(os.path.join(h5_dir,'*.xlsx')))
print(excel_files)

file_handle1 = excel_files[5]
print(file_handle1)
data_manual1 = pd.read_excel(file_handle1)

LoopLength = 1080
Variable = new_features.operculum
NumLoops = int(int(len(Variable))/LoopLength)

Manual1 = manual_scoring(data_manual1, Variable)

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

print(file_handle1)
#%%
Rates_df_manual = pd.DataFrame(Rates_manual)
ListA = [3,4,2,4,3,4,3,4,2,4,2,4,3,4,2,4,2,4,3,4,2,4]
Rates_df_manual['color'] = ListA

White_man = Rates_df_manual[Rates_df_manual['color'] == 4].mean()[0]
Blue_man = Rates_df_manual[Rates_df_manual['color'] == 2].mean()[0]
Red_man = Rates_df_manual[Rates_df_manual['color'] == 3].mean()[0]

print('White', White_man, 'Blue', Blue_man, 'Red' , Red_man)

# Rates_df.to_excel("907.xlsx")

#%%

index_man = (Blue_man - Red_man) /(Blue_man + Red_man)
print(index_man)

#%%
from sklearn.preprocessing import StandardScaler


Velocity = new_features.mov_speed
Comparison = pd.DataFrame(Variable)
Comparison['Manual1'] = Manual1
Comparison['Velocity'] = Velocity
x = StandardScaler().fit_transform(Comparison)

plt.plot(x)

#%%
#Create density plot of operculum opening 

LoopLength = 2160

NumLoops = int(int(len(Variable))/LoopLength)
Looped_Manual = pd.DataFrame(np.reshape(Manual1.values[:(LoopLength*NumLoops)],(NumLoops, LoopLength))).T

#%%
def uniform(indbin,LoopLength):
    '''
    Takes indbin
    '''
    A = len(indbin)
    B = LoopLength
    C = B - A
    end = np.zeros(C)
    new = np.concatenate((indbin, end))
    
    return(new)

def binarizeOp(Operangle, threshold):
    

    boolean = Operangle.apply(lambda x: 1 if x > threshold else 0).values
#    boolean = binary_dilation(boolean, structure = np.ones(40,))
    
    binindex = np.where(boolean)[0]

    return (binindex)


#%%
indbins = np.random.rand(0,LoopLength)
#
for i in np.arange(0,len(Looped.columns)):
    indbin = uniform(binarizeOp(Looped_Manual[i], threshold = 0), LoopLength)
    indbins = np.vstack([indbins, indbin])

#%%
fix, ax = plt.subplots(2,1, sharex = True)
ax[0].eventplot(indbins)

ax[1].set_xlim([0,2160])

A = sns.distplot(indbins[np.where(indbins != 0)], bins = 100, ax = ax[1])

plt.axvline(x = 1080, color = 'r')

#%%
# Check video for accuracy
fps = 40 
loop_id = 4
LoopLength = 2160
Frame_start = loop_id * LoopLength
Frame_stop = Frame_start + LoopLength
List_find = [Frame_start, Frame_stop]
for i in List_find:
    Time = (i/fps)/60
    Time_min = int(Time)
    Time_sec = (Time % 1)*60
    print(Time_min, Time_sec)
    
