#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:33:49 2020

@author: Claire
"""

# script for filtering and scoring automatic data, exports a excel file that includes:
# operangle overall rate, total velocity, position, orientation 

# exports an excel that gives binary of operculum events, the velocity, the x_position, orientation


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns; sns.set()
from helper_functions import *
import pickle
import ntpath

#%%

# set manual directories and select data
h5_dir = '/Users/Claire/Desktop/PiColor/Color_Jane/Make_mp4/'
csv_dir = '/Users/Claire/Desktop/PiColor/Alternating_Videos/2min_break_2_loop/'

# h5_file
h5_files = sorted(glob(os.path.join(h5_dir,'*.h5')))
file_handle1 = h5_files[0]
print(h5_files, file_handle1)

with pd.HDFStore(file_handle1,'r') as help1:
   data_auto1 = help1.get('df_with_missing')
   data_auto1.columns= data_auto1.columns.droplevel()

# csv file of stimulus code, only if alternating stim- not applicable for yuyang
# stim_seq_df = pd.read_excel(os.path.join(csv_dir, '1.4.2.xlsx'), header = 0).T
# stim_seq_df.columns = [ 'vid-code']
# stim_seq = stim_seq_df['vid-code']
# len_white = 60*2*40
# len_zero = 37*40
# len_one = 37*40

# index_repeat = list(map(lambda x:len_white if x==3 else x,stim_seq_df['vid-code']))
# index_repeat = list(map(lambda x:len_zero  if x==0 else x,index_repeat))
# index_repeat = list(map(lambda x:len_one if x==1 else x,index_repeat))
# print(index_repeat) 

# stim_seq = stim_seq.repeat(index_repeat)
#%%

# decode the video and set start/stop times for data analysis

hab_start = 0
test_start = 70000
test_end = 150000

#%%

# filter the data file 
new_features=features(starttime=hab_start, duration=len(data_auto1))

filtered_df=new_features.filter_df(data_auto1, add_midpoint = True)

new_features.fit(filtered_df,filter_feature=True,fill_na=True,estimate_na=False)

#%%

#extract out the operculum angle, orientation, velocity, position(x)

# Extract out the features for habituation time
Oper_Angle_Hab = np.array(new_features.operculum[hab_start:test_start])
Position_X_Hab =  np.array(filtered_df['A_head']['x'][hab_start:test_start] )#position has been filtered but not reinterpolated
Orientation_Hab = np.array( new_features.ori[hab_start:test_start])
Speed_Hab =  np.array(new_features.mov_speed[hab_start:test_start])
# Weighted_PosX_Ori_Hab =  np.array(((Orientation_Hab/np.linalg.norm(Orientation_Hab)*Position_X_Hab)).fillna(method='ffill')[hab_start:test_start])
# 
Oper_Angle_Test =  new_features.operculum[test_start:test_end]
Oper_Angle_Test_array =  np.array(new_features.operculum[test_start:test_end])
Position_X_Test = np.array(filtered_df['A_head']['x'][test_start:test_end]) #position has been filtered but not reinterpolated
Orientation_Test = np.array(new_features.ori[test_start:test_end])
Speed_Test = np.array(new_features.mov_speed[test_start:test_end])
# Weighted_PosX_Ori_Test = np.array((Orientation_Test/np.linalg.norm(Orientation_Test)*Position_X_Test)).fillna(method='ffill')[test_start:test_end])
Oper_Angle_Binary = np.array(binarize_Op_2(Oper_Angle_Test, lb = 65, ub = 135))



#%%

#customize the thresholding

# bin the data and show graphs for data split into 1080 windows
LoopLength = 1080
NumLoops = int(int(len(Oper_Angle_Test[20000:60000]))/LoopLength)
Looped = pd.DataFrame(np.reshape(Oper_Angle_Test[20000:60000].values[:(LoopLength*NumLoops)],(NumLoops, LoopLength))).T

custom_thresh_me = custom_thresh(Looped)


#%%
# Calculate Totals and Averages

if len(Oper_Angle_Hab)>0:
    Oper_Angle_Total = binarize_Op_2(Oper_Angle_Test, lb = 65, ub = 135).sum()/len(Oper_Angle_Test) - (binarize_Op_2(Oper_Angle_Hab, lb = 65, ub = 135).sum()/len(Oper_Angle_Hab))
    Position_X_Total = binarize_Op_2(Position_X_Test, lb = 0, ub = 100).sum()/len(Position_X_Test) - (binarize_Op_2(Position_X_Hab, lb = 0, ub = 100).sum()/len(Position_X_Hab))
    Orientation_Total = binarize_Op_2(Orientation_Test, lb = 90, ub = 180).sum()/len(Orientation_Test) - (binarize_Op_2(Orientation_Hab, lb = 90, ub = 180).sum()/len(Orientation_Hab))
    Speed_avg = np.average(Speed_Test) - (np.average(Speed_Hab))
    Oper_Angle_Total_custom = binarize_Op_2(Oper_Angle_Test, lb = custom_thresh_me, ub = 135).sum()/len(Oper_Angle_Test) - (binarize_Op_2(Oper_Angle_Hab, lb = custom_thresh_me, ub = 135).sum()/len(Oper_Angle_Hab))
    # Oper_Angle_Binary = binarize_Op_2(Oper_Angle_Test, lb = custom_thresh_me, ub = 135)
    
    # Weighted_Pos_Total = binarize_Op_2(Weighted_PosX_Ori_Test, +lb = 4, ub = Weighted_PosX_Ori_Test.max()).sum()/len(Position_X_Hab) - (binarize_Op_2(Weighted_PosX_Ori_Hab, lb = 4, ub = Weighted_PosX_Ori_Hab.max()).sum()/len(Position_X_Hab))

else:
    Oper_Angle_Total = binarize_Op_2(Oper_Angle_Test, lb = 65, ub = 135).sum()/len(Oper_Angle_Test) 
    Position_X_Total = binarize_Op_2(Position_X_Test, lb = 0, ub = 100).sum()/len(Position_X_Test) 
    Orientation_Total = binarize_Op_2(Orientation_Test, lb = 90, ub = 180).sum()/len(Orientation_Test) 
    Speed_avg = np.average(Speed_Test) - (np.average(Speed_Hab))
    Oper_Angle_Total_custom = binarize_Op_2(Oper_Angle_Test, lb = custom_thresh_me, ub = 135).sum()/len(Oper_Angle_Test) 
    Oper_Angle_Binary_custom = binarize_Op_2(Oper_Angle_Test, lb = custom_thresh_me, ub = 135)
    # Weighted_Pos_Total = binarize_Op_2(Weighted_PosX_Ori_Test, lb = 4, ub = Weighted_PosX_Ori_Test.max()).sum()/len(Position_X_Hab) - (binarize_Op_2(Weighted_PosX_Ori_Hab, lb = 4, ub = Weighted_PosX_Ori_Hab.max()).sum()/len(Position_X_Hab))

#%%
# load the tailbeating features


with open("IM1_IM2_2.1.1_L_70000_150000_tailAngle", "rb") as fp:  
    tail_Angle_array = pickle.load(fp)  
    
with open("IM1_IM2_2.1.1_L_70000_150000_curve_scores", "rb") as fp:  
    curve_scores_array = pickle.load(fp) 
    
with open("IM1_IM2_2.1.1_L_70000_150000_distance_to_spineline", "rb") as fp:  
    distance_to_spine_array = pickle.load(fp) 
    
with open("IM1_IM2_2.1.1_L_70000_150000_fish_segment_length", "rb") as fp:  
    segment_length_array = pickle.load(fp) 
    
with open("IM1_IM2_2.1.1_L_70000_150000_head_index", "rb") as fp:  
    head_indexes_array = pickle.load(fp) 

with open("IM1_IM2_2.1.1_L_70000_150000_tail_index", "rb") as fp:  
    tail_indexes_array = pickle.load(fp) 
    

#%%
plt.plot(Oper_Angle_Test_array, alpha=0.5)
plt.plot(Position_X_Test, alpha = 0.5)
plt.plot(Orientation_Test ,alpha = 0.5)
plt.plot(Speed_Test)
plt.plot(tail_Angle_array, alpha = 0.5)
# plt.plot(curve_scores_array[0], alpha = 0.5)
plt.plot(distance_to_spine_array, alpha = 0.5)
# plt.plot(segment_length_array, alpha = 0.5)



#%%

Data_Concat = pd.DataFrame()
Data_Concat['Oper_Angle'] = Oper_Angle_Test
Data_Concat['X_Position'] = Position_X_Test
Data_Concat['Orientation'] = Orientation_Test
Data_Concat['Speed'] = Speed_Test
Data_Concat['Tail_Angle'] = tail_Angle_array
Data_Concat['Tail_Deviation'] = distance_to_spine_array
 #create dataframes and export as .exlsx
export_dir = '/Users/Claire/Desktop/PiColor'
Data_Concat.to_excel(os.path.join(export_dir, ntpath.basename(file_handle1))+ ".xlsx")
# export all continuous features 



