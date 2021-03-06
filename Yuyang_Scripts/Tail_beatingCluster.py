#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:54:25 2020

@author: ryan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns; sns.set()
#from helper_functions import *
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
import importlib.machinery
from sklearn.cluster import MeanShift,SpectralClustering  
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from hmmlearn import hmm
from sklearn.metrics import f1_score
from tqdm import tqdm
from find_features import features
from contour_utils import relative_position_check
from functions import speed
import pickle
IMAGELENGTH=500
#load module from Claire's script
loader = importlib.machinery.SourceFileLoader('helper_functions', os.path.join(os.path.abspath('..'),"Claire_scripts",'helper_functions.py'))
helper_functions = loader.load_module('helper_functions')


data_auto_scored = pd.read_excel("Claire_score_tables/IM1_IM2_auto_data_scored.xlsx")
data_auto_scored.rename(columns={"Unnamed: 0":"time"},inplace=True)
data_manual = pd.read_excel("Claire_score_tables/IM1_IM2_L_Manual_score.xlsx")

path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
data_auto1=df
starttime = 90000
endtime = 140000
Manual1 = np.array(helper_functions.manual_scoring(data_manual,data_auto1, crop0 = starttime, crop1 = endtime))

data_auto_scored_trunc=data_auto_scored[20000:70000]

del data_auto_scored_trunc["time"]
#maybe should not delete?
del data_auto_scored_trunc['X_Position']

data_auto_scored_trunc["diff_tail_angle"]=np.diff(data_auto_scored_trunc.Tail_Angle,prepend=data_auto_scored_trunc.Tail_Angle.iloc[0])
data_auto_scored_trunc["diff_tail_dev"]=np.diff(data_auto_scored_trunc.Tail_Deviation,prepend=data_auto_scored_trunc.Tail_Deviation.iloc[0])


data=data_auto_scored_trunc.drop(["Tail_Angle","Tail_Deviation"],axis=1)
data=pd.DataFrame(StandardScaler().fit_transform(data),columns=data.columns)
data=data.iloc[::20,:]
Manual=Manual1[::20]

#check if one of the cluster captures the tail beating manual
for i in range(2,9):
    spectral=SpectralClustering(n_clusters=i).fit(data)
    scores=[]
    labels=spectral.labels_
    for l in range(spectral.n_clusters):
        projected_labels=np.where(labels==l,1,0)
        scores.append(adjusted_rand_score(Manual,projected_labels))
    print(np.max(np.array(scores)))

#so the cluster result is not that good, the best num of k is 3, but still gets a very poor result

for i in range(2,10):
    gaussianHMM=hmm.GaussianHMM(n_components=i,covariance_type="full").fit(data)
    scores=[]
    labels=gaussianHMM.predict(data)
    for l in range(i):
        projected_labels=np.where(labels==l,1,0)
        scores.append(f1_score(Manual1,projected_labels))
    print(np.max(scores))

#using a time series makes better, the score reaches around 0.3

colors=[]
for i in Manual1:
    if i==0:
        colors.append("yellow")
    else:
        colors.append("red")
plt.scatter(data['Oper_Angle'],data['diff_tail_dev'],c=colors)
plt.xlabel("Operculum Angle")
plt.ylabel("difference in Tail deviance")

n_components=9
gaussianHMM=hmm.GaussianHMM(n_components=n_components,covariance_type="full").fit(data)
scores=[]
labels=gaussianHMM.predict(data)
for l in range(n_components):
    projected_labels=np.where(labels==l,1,0)
    scores.append(f1_score(Manual1,projected_labels))
plt.scatter(data['Oper_Angle'],data['diff_tail_dev'],c=labels)
plt.xlabel("Operculum Angle")
plt.ylabel("difference in Tail deviance")

'''
plt.scatter(data['Oper_Angle'],data['diff_tail_dev'],c=colors)
plt.xlabel("Operculum Angle")
plt.ylabel("difference in Tail deviance")

colors=[]
for i in labels:
    if i==1:
        colors.append("yellow")
    else:
        colors.append("red")
 '''

plt.scatter(data['Oper_Angle'],data['diff_tail_angle'],c=labels)
plt.xlabel("Operculum Angle")
plt.ylabel("difference in Tail Angle")

plt.scatter(data['Oper_Angle'],data['diff_tail_angle'],c=colors)
plt.xlabel("Operculum Angle")
plt.ylabel("difference in Tail Angle")

pd.DataFrame(gaussianHMM.means_,columns=data.columns)
#so it's like the clustering is including more points than the actual tail beating behavior, but it's reasonable now? given how it is actually defined

#to do:
#there seems to be a cluster where the diff_tail_dev being extremely large ,which is werid
#try some feature selection? not sure what to do yet, maybe find feature importance first by 

#%%
'''
My plan: 1.First look at the fourier transformations of each tail beating period, try to see certain patterns in it
1.5 Maybe remove some insignificant noises.
2.Then do a stft on the whole period, then for each short segment, computed the weighted sum of the scaled amplitude(\sum freq X scaled amplitude),
since higher frequency should have a higher portion if the fish is waggling its tail rapidly
3.use this as feature
'''
def window_function(x,minimal=1.5,maximal=5):
    return np.where(np.logical_and(x>minimal,x<maximal),x*x,0)

#try fourier transformation on each tail beating period
from scipy.fft import fft,ifft,fftfreq
fps=40

n_period=data_manual.shape[0]
weighted_freqs=[]
#for each period, plot the amplitude vs frequency
#also computed the weighted sum of the scaled amplitude for each period
#I can't physically justify what this measure means, but from intuition it should have a higher value when fish is waggling its tail

for i in range(n_period):
    start=data_manual.iloc[i,0]
    stop=data_manual.iloc[i,1]
    #the tail_angle in that certain period
    tail_beating_period=np.array(data_auto_scored[np.logical_and(data_auto_scored['time']>=start,data_auto_scored['time']<=stop)]
                                 .Tail_Angle)
    freqy=fft(tail_beating_period)
    #the corresponding frequencies
    period_length=len(tail_beating_period)
    freqs=fftfreq(period_length,d=1/fps)
    #take only positive frequencies as it is symmetric
    freqs=freqs[1:(period_length-1)//2+1]
    freqy=freqy[1:(period_length-1)//2+1]
    weighted_freq=np.sum(np.abs(freqy)/np.sum(np.abs(freqy))*window_function(freqs))
    weighted_freqs.append(weighted_freq)
    plt.figure()
    plt.plot(freqs,np.abs(freqy))
    plt.xlim(0,np.max(freqs))
    plt.xlabel("frequency(HZ)")
    plt.ylabel("amplitude")
    plt.title("FFT for the {}_th period".format(i+1))

#check some other period 
#stationary period
non_beating_period1=np.array(data_auto_scored[np.logical_and(data_auto_scored['time']>=93220,data_auto_scored['time']<=93320)]
                                 .Tail_Angle)
freqy=fft(non_beating_period1)
    #the corresponding frequencies
period_length=len(non_beating_period1)
freqs=fftfreq(period_length,d=1/fps)
#take only positive frequencies as it is symmetric
freqs=freqs[1:(period_length-1)//2+1]
freqy=freqy[1:(period_length-1)//2+1]
print("the weighted frequency is {}".format(np.sum(np.abs(freqy)/np.sum(np.abs(freqy))*window_function(freqs))))
plt.figure()
plt.plot(freqs,np.abs(freqy))
plt.xlim(0,np.max(freqs))
plt.xlabel("frequency(HZ)")
plt.ylabel("amplitude")
plt.title("FFT when fish facing east1".format(i+1))

#fish is making a turn
non_beating_period2=np.array(data_auto_scored[np.logical_and(data_auto_scored['time']>=93360,data_auto_scored['time']<=93440)]
                                 .Tail_Angle)
freqy=fft(non_beating_period2)
    #the corresponding frequencies
period_length=len(non_beating_period2)
freqs=fftfreq(period_length,d=1/fps)
#take only positive frequencies as it is symmetric
freqs=freqs[1:(period_length-1)//2+1]
freqy=freqy[1:(period_length-1)//2+1]
print("the weighted frequency is {}".format(np.sum(np.abs(freqy)/np.sum(np.abs(freqy))*window_function(freqs))))
plt.figure()
plt.plot(freqs,np.abs(freqy))
plt.xlim(0,np.max(freqs))
plt.xlabel("frequency(HZ)")
plt.ylabel("amplitude")
plt.title("FFT when fish is making a turn".format(i+1))

#another stationary period
non_beating_period3=np.array(data_auto_scored[np.logical_and(data_auto_scored['time']>=101870,data_auto_scored['time']<=101950)]
                                 .Tail_Angle)
freqy=fft(non_beating_period3)
    #the corresponding frequencies
period_length=len(non_beating_period3)
freqs=fftfreq(period_length,d=1/fps)
#take only positive frequencies as it is symmetric
freqs=freqs[1:(period_length-1)//2+1]
freqy=freqy[1:(period_length-1)//2+1]
print("the weighted frequency is {}".format(np.sum(np.abs(freqy)/np.sum(np.abs(freqy))*window_function(freqs))))
plt.figure()
plt.plot(freqs,np.abs(freqy))
plt.xlim(0,np.max(freqs))
plt.xlabel("frequency(HZ)")
plt.ylabel("amplitude")
plt.title("FFT when fish is facing east2".format(i+1))

non_beating_period4=np.array(data_auto_scored[np.logical_and(data_auto_scored['time']>=101960,data_auto_scored['time']<=102000)]
                                 .Tail_Angle)
freqy=fft(non_beating_period4)
    #the corresponding frequencies
period_length=len(non_beating_period4)
freqs=fftfreq(period_length,d=1/fps)
#take only positive frequencies as it is symmetric
freqs=freqs[1:(period_length-1)//2+1]
freqy=freqy[1:(period_length-1)//2+1]
print("the weighted frequency is {}".format(np.sum(np.abs(freqy)/np.sum(np.abs(freqy))*window_function(freqs))))
plt.figure()
plt.plot(freqs,np.abs(freqy))
plt.xlim(0,np.max(freqs))
plt.xlabel("frequency(HZ)")
plt.ylabel("amplitude")
plt.title("FFT when fish is making a turn 2".format(i+1))
#%%
#test stft on the entire period
tail_angle=np.array(data_auto_scored.Tail_Angle)
f, t, Zxx = signal.stft(tail_angle, fs=40,nperseg=100,window = "hann", noverlap=50)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=2, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()

weighted_freqs_stft=np.matmul(window_function(f).reshape(1,51),np.abs(Zxx)/np.sum(np.abs(Zxx),axis=0)[None,:]).reshape(1601)
def stft_freqs2feature(x):
    l=len(x)
    output=np.array([])
    for i in range(l):
        #broadcast the value to the neighbors of that certain time.
        #since segment center is picked by 0,50,100...,so each point extend 25 points to both left and right
        if i==0 or i==l-1:
            output=np.concatenate((output,np.repeat(x[i],25)))
        else:
            output=np.concatenate((output,np.repeat(x[i],50)))
    return output
    
freq_feature=stft_freqs2feature(weighted_freqs_stft)  

#visualize 90000-100000th frame   
freq_feature_trunc=freq_feature[20000:30000]
manual_trunc=Manual1[:10000]
labels_trunc=labels[:10000]
import cv2 
# path 
path = "TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"
cap = cv2.VideoCapture(path) 
index=90000
colormap=[(255,0,0), (0,0,0), (51,255,51),(51,51,255),(51,255,255),(255,128,0),(255,255,255),(255,255,0),(204,0,204)]
cap.set(1,index)
result = cv2.VideoWriter('videos/test_tailbeatcluster.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGELENGTH,IMAGELENGTH))
for i in range(10000): 
    # Capture frames in the video 
    ret, frame = cap.read() 
    if ret!=True:
        break
    label=labels_trunc[i]
    feature=freq_feature_trunc[i]
    manual=manual_trunc[i]
    color=colormap[label]
    manual_color= (0,0,0) if manual==1 else (255,255,255)
    # describe the type of font 
    # to be used. 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #put on the predicted cluster
    frame=cv2.circle(frame, (50,50), 50, color, -1)
    frame=cv2.putText(frame,  
                "predicted phase, {}".format(label),  
                (0, 110),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    #put on the manual one
    frame=cv2.circle(frame, (50,180), 50, manual_color, -1)
    frame=cv2.putText(frame,  
                "manual phase, {}".format(manual),  
                (0, 240),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    #write the frequency feature
    frame=cv2.putText(frame,  
                "frequency feature = {}".format(feature),  
                (0, 450),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    result.write(frame)
result.release()
  

#%%
#Try to select optimal features here

with open("data/IM1_IM2_2.1.1_L_70000_150000_orientations", "rb") as fp:  
    orientations = pickle.load(fp)
with open("data/IM1_IM2_2.1.1_L_70000_150000_velocity", "rb") as fp:  
    velocity = pickle.load(fp)
with open("data/IM1_IM2_2.1.1_L_70000_150000_tailAngle_wsign", "rb") as fp:  
    signed_angle = pickle.load(fp)
with open("data/IM1_IM2_2.1.1_L_70000_150000_distance_to_spineline_wsign", "rb") as fp:  
    signed_dev = pickle.load(fp)


data_auto_scored_trunc=data_auto_scored[20000:70000]
del data_auto_scored_trunc["time"]
data=data_auto_scored_trunc
data['tail_beat_freq']=freq_feature[20000:70000]
data["Orientation"]=np.array(orientations)[20000:70000]
data["Speed"]=velocity[20000:70000]
data['signed_angle']=signed_angle[20000:70000]
data["signed_dev"]=signed_dev[20000:70000]
data["diff_tail_angle"]=np.diff(data.signed_angle,prepend=data.signed_angle.iloc[0])
data["diff_tail_dev"]=np.diff(data.signed_dev,prepend=data.signed_dev.iloc[0])
del data['signed_angle']
del data['signed_dev']
data["abs_diff_angle"]=np.abs(data.diff_tail_angle)
data["abs_diff_dev"]=np.abs(data.diff_tail_dev)
scaler=StandardScaler()
data=pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
data=data.fillna(method="ffill")


#operculum and orientation is definitly important, so I don't select them
must_include=["Oper_Angle","Orientation","X_Position","tail_beat_freq"]
debatable=["Speed","Tail_Angle","Tail_Deviation","diff_tail_angle","diff_tail_dev","abs_diff_angle","abs_diff_dev"]
def all_possible_features(x):
    '''
    select all possible subsets for a given set
    '''
    if len(x)==1:
        return [x,[]]
    #recursion,find all possible sets excluding x[0], then add x[0] to each of them
    output=[]
    first_element=x[0]
    other_possible_sets=all_possible_features(x[1:])
    for s in other_possible_sets:
        output.append(s)
        output.append(s+[first_element])
    #add the must-include features
    return output

all_possible_sets=all_possible_features(debatable)
for i in range(len(all_possible_sets)):
    all_possible_sets[i]+=must_include

#for each feature, run HMM 2 times for each num of cluster
best_result=[]
#store the best result for each selected feature, e.g. {["Oper","speed"]:[0.35 (f1_score),8(num_components)]}
#do not run this!!
for s in tqdm(all_possible_sets):
    temp_data=data[s]
    feature_score={}
    l=len(s)
    for i in range(3+int(l/2),6+l):
        meta_scores=[]
        #run HMM 2 times for each n_component
        for j in range(2):
            gaussianHMM=hmm.GaussianHMM(n_components=i,covariance_type="full").fit(temp_data)
            scores=[]
            labels=gaussianHMM.predict(temp_data)
            for l in range(i):
                projected_labels=np.where(labels==l,1,0)
                scores.append(f1_score(Manual1,projected_labels))
            meta_scores.append(np.max(scores))
        #append avg score for each num_cluster
        avg=np.mean(np.array(meta_scores))
        feature_score[i]=avg
    #append best result for each set of features
    key=max(feature_score,key=feature_score.get)
    best_result.append([s,key,feature_score[key]])

import pickle 
result=sorted(best_result,key=lambda x:x[2],reverse=True)

test=sorted(best_result,key=lambda x:1 if set(x[0])==set(must_include+["abs_diff_tail_dev"]) else 0,reverse=True)
#the result of all possible feature sets in a descending order of performance
with open("feature_scores_updated", 'wb') as f: 
    pickle.dump(result, f)  
    
s=result[0][0]



#%%below is the same video making process
temp_data=data[s]
n_components=5
#run HMM on the optimal feature again
gaussianHMM=hmm.GaussianHMM(n_components=n_components,covariance_type="full").fit(temp_data)
with open("models/HMM_5states_5features.pkl","wb") as f:
    pickle.dump(gaussianHMM,f)
    
scores=[]
labels=gaussianHMM.predict(temp_data)
for l in range(n_components):
    projected_labels=np.where(labels==l,1,0)
    scores.append(f1_score(Manual1,projected_labels))

#
freq_feature_trunc=freq_feature[20000:30000]
max_freq_feature=np.max(freq_feature)
manual_trunc=Manual1[:10000]
labels_trunc=labels[:10000]
import cv2 
# path 
path = "TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"
cap = cv2.VideoCapture(path) 
index=90000
colormap=[(255,0,0), (51,51,255), (51,255,51),(51,255,255),(0,0,0),(255,128,0),(255,255,255),(255,255,0),(204,0,204),(160,160,160)]
cap.set(1,index)
result = cv2.VideoWriter('videos/current_optimal_feature_cluster_updated.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGELENGTH,IMAGELENGTH))
for i in range(10000): 
    # Capture frames in the video 
    ret, frame = cap.read() 
    if ret!=True:
        break
    label=labels_trunc[i]
    feature=freq_feature_trunc[i]
    manual=manual_trunc[i]
    color=colormap[label]
    manual_color= (0,0,0) if manual==1 else (255,255,255)
    # describe the type of font 
    # to be used. 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #put on the predicted cluster
    frame=cv2.circle(frame, (50,50), 50, color, -1)
    frame=cv2.putText(frame,  
                "predicted phase, {}".format(label),  
                (0, 110),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    #put on the manual one
    frame=cv2.circle(frame, (50,180), 50, manual_color, -1)
    frame=cv2.putText(frame,  
                "manual phase, {}".format(manual),  
                (0, 240),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    #write the frequency feature
    frame=cv2.rectangle(frame,(0,425),(int(feature/max_freq_feature*200),438),(0,0,0),-1)
    frame=cv2.putText(frame,  
                "frequency feature = {}".format(feature),  
                (0, 450),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    frame=cv2.putText(frame,  
                "{} frame".format(i),  
                (0, 470),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    result.write(frame)
result.release()

##something else?
#look at AIC/BIC/other goodness of fit measures
#explaining the hidden states?
#abs of the dev?
#improve tail angle and tail dev ? tell which side it is on?
means=pd.DataFrame(gaussianHMM.means_,columns=s)
covars=gaussianHMM.covars_
covars_perCluster=[]
for covar in covars:
    covar_df=pd.DataFrame(covar,columns=s)
    covars_perCluster.append(covar_df)

#use histogram for each feature!!!!!!!!

#%%
#interpretations
'''
5 state, the simple one

hidden state0: - operculum angle, + X_position, - tail_beat_freq, avg Orientation: 
Inferred action: fish sticking to the right glass in a tilted position, with operculum closed, likely moving to north or south

hidden state1: + Operculum angle , - orientation, + X_position, + tail_beat_freq
covar: high var for tail_beat_freq and Oper_angle
Inferred Action: flaring+moving tail

hidden state2: + Operculum angle, - orientation, - tail_freq
Inferred Action: flaring but more stationary, possibly flaring in a distance, or entering/exiting flaring(advancing to east/retreating from east)

hidden state3: -Operculum Angle, +Orientation, - X_position
covar: but all features seem to have rather large variance.
Inferred Action: unclear, it's like the "leftover" state", a large proportion of it is when fish is facing north but not 
tail beating, but there are also other frames be put into state3 that don't satisfy condition of state 0,1,2

hidden state4: + diff_tail_dev, -- Operculum_angle, ++ Orientation, -- X_Position, + tail_beat_freq
covars: diff_tail_dev has a extremely large variance, X_position and tail_beat_freq also have a large var.
Inferred Action: tail beating


other possible actions:
    Idling: like when fish is not flaring nor moving its tail, but since it's a fighting period, the fish is not that idle
    turn: Looks like hard to define a turn. First it overlaps with some other states, like the fish instantly starts flaring
    when it turns its head to east. Secondly currently there are not many features to define the action.
'''

df=pd.DataFrame(scaler.inverse_transform(data),columns=data.columns)
for i in range(5):
    plt.hist(df.Oper_Angle[labels==i],label="hidden state{}".format(i),alpha=0.8,bins=15)
plt.legend()
plt.xlabel("degree")
plt.ylabel("count")
plt.title("Operculum distribution for each hidden state")

for i in range(5):
    plt.hist(df.diff_tail_dev[labels==i],label="hidden state{}".format(i),alpha=0.8,bins=40)
plt.legend()
plt.xlabel("frames")
plt.ylabel("count")
plt.xlim(-100,100)
plt.title("difference in tail deviation's distribution for each hidden state")

for i in range(5):
    plt.hist(df.Orientation[labels==i],label="hidden state{}".format(i),alpha=0.8,bins=20)
plt.legend()
plt.xlabel("degree")
plt.ylabel("count")
plt.title("Orientation distribution for each hidden state")

for i in range(5):
    plt.hist(df.X_Position[labels==i],label="hidden state{}".format(i),alpha=0.8,bins=20)
plt.legend()
plt.xlim(left=370)
plt.xlabel("frame")
plt.ylabel("count")
plt.title("X value's distribution for each hidden state")

for i in range(5):
    plt.hist(df.tail_beat_freq[labels==i],label="hidden state{}".format(i),alpha=0.8,bins=20)
plt.legend()
plt.xlabel("HZ^2")
plt.ylabel("count")
plt.title("frequancy feature distribution for each hidden state")

from scipy import signal
t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.colorbar()
plt.show()


points = 100
a = 4.0
vec2 = signal.ricker(points, 4)
freqy=fft(vec2)
freqs=fftfreq(100)
plt.plot(freqs[:50],np.abs(freqy)[:50])
freqs[np.argmax(np.abs(freqy)[:50])]

import pywt
t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
widths = np.arange(0.25, 10,0.25)
cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')

