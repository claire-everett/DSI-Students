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

loader = importlib.machinery.SourceFileLoader('helper_functions', os.path.join(os.path.abspath('..'),"Claire_scripts",'helper_functions.py'))
helper_functions = loader.load_module('helper_functions')



data_auto_scored = pd.read_excel("Claire_score_tables/IM1_IM2_auto_data_scored.xlsx")
data_manual = pd.read_excel("Claire_score_tables/IM1_IM2_L_Manual_score.xlsx")

path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
data_auto1=df
starttime = 90000
endtime = 140000
Manual1 = np.array(helper_functions.manual_scoring(data_manual,data_auto1, crop0 = starttime, crop1 = endtime))

data_auto_scored=data_auto_scored[20000:70000]
del data_auto_scored["Unnamed: 0"]
del data_auto_scored['X_Position']

data_auto_scored["diff_tail_angle"]=np.diff(data_auto_scored.Tail_Angle,prepend=data_auto_scored.Tail_Angle.iloc[0])
data_auto_scored["diff_tail_dev"]=np.diff(data_auto_scored.Tail_Deviation,prepend=data_auto_scored.Tail_Deviation.iloc[0])


data=data_auto_scored.drop(["Tail_Angle","Tail_Deviation"],axis=1)
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

data=data_auto_scored.drop(["Tail_Angle","Tail_Deviation"],axis=1)
data=pd.DataFrame(StandardScaler().fit_transform(data),columns=data.columns)
for i in range(2,10):
    gaussianHMM=hmm.GaussianHMM(n_components=i,covariance_type="full").fit(data)
    scores=[]
    labels=gaussianHMM.predict(data)
    for l in range(i):
        projected_labels=np.where(labels==l,1,0)
        scores.append(adjusted_rand_score(Manual1,projected_labels))
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

n_components=8
gaussianHMM=hmm.GaussianHMM(n_components=n_components,covariance_type="full").fit(data)
scores=[]
labels=gaussianHMM.predict(data)
for l in range(n_components):
    projected_labels=np.where(labels==l,1,0)
    scores.append(adjusted_rand_score(Manual1,projected_labels))
plt.scatter(data['Oper_Angle'],data['diff_tail_dev'],c=labels)
plt.xlabel("Operculum Angle")
plt.ylabel("difference in Tail deviance")

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





