#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:17:11 2020

@author: Claire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns; sns.set()
from helper_functions import *
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
#%%
## load the automated scoring features
excel_dir = '/Users/Claire/Desktop/PiColor/'
excel_files = sorted(glob(os.path.join(excel_dir,'*.xlsx')))
print(excel_files)

file_handle1 = excel_files[1]
print(file_handle1)
data_auto_scored = pd.read_excel(file_handle1)

## load the manual scoring data
excel_files = sorted(glob(os.path.join(excel_dir,'*.xlsx')))
print(excel_files)

file_handle1 = excel_files[2]
print(file_handle1)
data_manual = pd.read_excel(file_handle1)

## load the raw tracking
home_dir = '/Users/Claire/Desktop/PiColor/Color_Jane/Make_mp4/'#'/Users/Claire/Desktop/FishBehaviorAnalysis'
h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
file_handle1 = h5_files[0]
print(file_handle1)

with pd.HDFStore(file_handle1,'r') as help1:
   data_auto1 = help1.get('df_with_missing')
   data_auto1.columns= data_auto1.columns.droplevel()
#%%
starttime = data_auto_scored['Unnamed: 0'][0]
endtime = data_auto_scored['Unnamed: 0'][len(data_auto_scored)-1]
Manual1 = np.array(manual_scoring(data_manual,data_auto1, crop0 = starttime, crop1 = endtime+1))


#%%
data_auto_scored['Manual_Score'] = Manual1

Manual_TB = data_auto_scored[data_auto_scored['Manual_Score'] == 1]

Manual_TB.index = Manual_TB['Unnamed: 0']
#%%

# shows the tail angle when the manual scoring says the fish is tail beating
for i in np.arange(len(data_manual)):
    new = np.array(Manual_TB['Tail_Angle'][Manual_TB['Unnamed: 0'].loc[data_manual.values[i][0]:data_manual.values[i][1]]])
    plt.plot(new)
    plt.title(i)
    plt.show()
    
#%%


# Continuous Short time Fourier Transform 

# f, t, Zxx = signal.stft(Manual_TB['Tail_Angle'][Manual_TB['Unnamed: 0'].loc[data_manual.values[3][0]:data_manual.values[3][1]]], window = "hann", nperseg=40)
f, t, Zxx = signal.stft(Manual_TB['Tail_Angle'], window = "hann", noverlap=50)

# f, t, Zxx = signal.stft(data_auto_scored['Tail_Angle'], window = 'hann',  nperseg=40 )

amp = 2 * np.sqrt(1)
# zxx_df = pd.DataFrame(Zxx)
# Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
# f = np.where(np.abs(f) <= 0.15, f, 0)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
# for i in np.arange(len(data_manual)):
#     plt.axvline(x = data_manual.values[i][0]-70000, color='g', alpha = 0.3)
#     plt.axvline(x = data_manual.values[i][1]-70000, color= 'g', alpha = 0.3)
#     # plt.axvspan(data_manual.values[i][0]-70000, data_manual.values[i][1]-70000, facecolor='b', alpha=1)
plt.show()
# for i in np.arange(len(Zxx)):
#     plt.plot(t, Zxx[i])
#     for i in np.arange(len(data_manual)):
#         plt.axvline(x = data_manual.values[i][0]-70000, color='r', alpha = 0.3)
#         plt.axvline(x = data_manual.values[i][1]-70000, color= 'y', alpha = 0.3)
#     plt.show()

  

#%%
# makes a video of select frames

cap = cv2.VideoCapture('IM1_IM2_2.1.1_L_TS.mp4')

# frameIds = np.array(Manual_TB['Unnamed: 0'])
frameIds = np.arange(data_manual.values[17][0], data_manual.values[17][1])

frames = []

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = cv2.VideoWriter('Select_good.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

for fid in frames:
    out.write(fid)
    
#%%

##SCRAP**** still working on any code below


#fourier transform

Zxx_2 = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
_, xrec = signal.istft(Zxx_2, fs)

#%%

#%%
# try to make a conditional 
#data_auto_scored['stft_score'] = # make a column for if the tail is moving at a certain frequency of a certain power that coinsides with the tail beating without doubt

#%%
# Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)

# _, xrec = signal.istft(Zxx, fs)

#%%

# plt.plot(t, x, t, xrec)
#%%


f, Pxx_den = signal.welch(Manual_TB['Tail_Angle'][0:47], fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
plt.show()


## learn how this all works, what is the Zxx, what does it consist of, can I threshold it, if I can, how generalizable is it going to be?
# make a number of conditions, if the fish is still, if they are oriented north or south, and if falls within a certain frequency- tail beating

#%%
time = np.arange(250)
Zxx = np.where(np.abs(Zxx) >= amp/60, Zxx, 0)
_, xrec = signal.istft(Zxx, fs = 1.0, window = "hann", noverlap=50)
# plt.plot(time, Manual_TB['Tail_Angle'][:250], time, xrec[:250])
plt.plot(time, xrec[:250])

#%%
from sklearn.manifold import TSNE

Test = data_auto_scored[data_auto_scored.columns[1:5]]
Test = Test.fillna(method="ffill")

t = StandardScaler().fit_transform(Zxx)

# X_embedded = TSNE(n_components=2).fit_transform(x)


pca = PCA()
pca.fit(t)
pcs=pca.transform(t)


pcs=pcs[:,:2]
pcs = np.clip(pcs,-3,3)
plt.scatter(pcs[:,0],pcs[:,1],s = 1) 
    
    
