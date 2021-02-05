#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 12:56:15 2021

@author: ryan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sn
from pipeline_functions import *
#from helper_functions import *
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
import os
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.metrics import f1_score
from tqdm import tqdm
import pickle
from scipy.signal import argrelextrema,find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
IMAGELENGTH=500
fps=40
NUM_STATE = 6

#make a series of figures on the predicted/true label
#log likelihood for each hidden states/feature selection
'''
For the trained model check tail beating and flaring distribution in states

Train on all video, get the pose feature.
 
take the last 10000 frame from each fish as test data, Run Glm with and without pose features from HMM,
other features are PCA of the across fish features and the fish feature from opponent fish.
'''
with open('models/GaussianHMM_6states_Jan18.pkl',"rb") as fp:
    hmm_model = pickle.load(fp)
    
#get the frequency feature
def get_freq_feature(data):
   tail_angle=np.array(data.Tail_Angle)
   tail_angle=np.array(pd.Series(tail_angle).fillna(method="ffill"))
   #stft transform
   f, t, Zxx = signal.stft(tail_angle, fs=40,nperseg=70,window = "hann", noverlap=69,boundary=None)
   Zxx_abs=np.abs(Zxx)
   scaled_Zxx=Zxx_abs[2:]/np.max(Zxx_abs[2:],axis=0)[None,:]
   
   #f<2 and f>10 truncated to 0
   #f in [5,10] remains at 5
   f_weight=f.copy()
   f_weight[f_weight>10]=0
   f_weight[f_weight>5]=5
   f_weight[f_weight<=2]=0

   freq_feature=[]
   for i in tqdm(range(Zxx_abs.shape[1])):
    local_max=find_peaks(scaled_Zxx[:,i],distance=3,height=0.2)[0]
    max_amplitude=np.max(Zxx_abs[np.logical_and(f>2,f<10),i])
    #amplitude scaling factor
    amplitude_scaler=float(min(1,max_amplitude/3)**2)
    if len(local_max)==0:
        freq_feature.append(0)
    else:
        #maximum of [frequency at local maximum * amplitude]
        score=np.max(f_weight[2:][local_max]*scaled_Zxx[:,i][local_max])*amplitude_scaler
        freq_feature.append(score)
   freq_feature=np.concatenate([np.repeat(np.nan,35),np.array(freq_feature),np.repeat(np.nan,34)])
   return freq_feature


def manual_scoring(data_manual,length,crop0 = 0,crop1= -1):
    '''
    A function that takes manually scored data and converts it to a binary array. 
    
    Parameters: 
    data_manual: manual scored data, read in from an excel file
    data_auto: automatically scored data, just used to establish how long the session is. 
    
    Returns: 
    pandas array: binary array of open/closed scoring
    '''
    Manual = pd.DataFrame(0, index=np.arange(length), columns = ['OpOpen'])
    reference = data_manual.index
    
    
    for i in reference:
        Manual[data_manual['Start'][i]:data_manual['Stop'][i]] = 1
    
    return Manual['OpOpen'][crop0:crop1]


#files=sorted(glob("data/*.csv"))
feature_columns=['operculum', 'orientation', 'movement_speed', 'turning_angle',
       'Tail_Angle', 'Tail_Dev', 'X_Position', 'curviness', 'freq_feature']
 
fish_names = ["IM1_IM2_L", "IM1_IM2_R", "VM3_VM4_L", "IM2_IM4_R",
              "VM1_VM3_L", "VM1_VM3_R", "IM2_IM4_L" , "VM3_VM4_R"]

sheetnames=["IM1_IM2_L","IM1_IM2_R","VM3_VM4_L","IM2_IM4_R","VM1_VM3_L","VM1_VM3_R","IM2_IM4_L","VM3_VM4_R"]
xls = pd.ExcelFile('TailBeatingExamples/TailManual(1).xlsx')

#split train and test data, compute tail frequency
train_data=pd.DataFrame()
test_data=pd.DataFrame()
train_set=[]
test_set=[]
train_tail_manual=[]
test_tail_manual=[]
for i, path in enumerate(fish_names):
    file_path = os.path.join("data", path, "*.csv")
    file = glob(file_path)[0]
    data=pd.read_csv(file)
    freq_feature=get_freq_feature(data)
    data["freq_feature"]=freq_feature
    data=data.iloc[20000:70000,:]
    manual = pd.read_excel(xls, sheetnames[i])
    manual = np.array(manual_scoring(manual,200000, crop0 = 90000, crop1 = 140000))
    if "R" in file: #flip Right images to Left
        data['X_Position']=500-data['X_Position']
        data["orientation"]=180-data["orientation"]
        data["orientation_from_contour"]=180-data["orientation_from_contour"]
    if i<6:
        train_data=train_data.append(data)
        train_set.append(file)
        train_tail_manual.append(manual)
        #create np.nan filled buffer if neccessary
        if i!=5:#last data in train
            buffer=np.full((100,len(feature_columns)),np.nan)
            train_data=train_data.append(pd.DataFrame(buffer,columns=feature_columns))
            train_tail_manual.append(np.zeros(100))
    else:
        test_data=test_data.append(data)
        test_set.append(file)
        test_tail_manual.append(manual)
        if i!=7:#last data in test
            buffer=np.full((100,len(feature_columns)),np.nan)
            test_data=test_data.append(pd.DataFrame(buffer,columns=feature_columns))
            test_tail_manual.append(np.zeros(100))
            
#train_data.reset_index(inplace=True)
#test_data.reset_index(inplace=True)

train_tail_manual=np.concatenate(train_tail_manual)
test_tail_manual=np.concatenate(test_tail_manual)

train_data.interpolate(method="linear",inplace=True)
test_data.interpolate(method="linear",inplace=True)

train_data["diff_tail_dev"]=np.diff(train_data.Tail_Dev,prepend=train_data.Tail_Dev.iloc[0])
test_data["diff_tail_dev"]=np.diff(test_data.Tail_Dev,prepend=test_data.Tail_Dev.iloc[0])

train_tailbeat_manual = train_tail_manual
test_tailbeat_manual = test_tail_manual
manual_flaring = pd.ExcelFile('TailBeatingExamples/Flaring_yuyang-2.xlsx')
train_flaring_manual=[]
test_flaring_manual=[]
for i in range(len(fish_names)):
    manual = pd.read_excel(manual_flaring, sheetnames[i])
    manual = np.array(manual_scoring(manual,200000, crop0 = 90000, crop1 = 140000))
    if i < 6:
        train_flaring_manual.append(manual)
        if i != 5:
            train_flaring_manual.append(np.zeros(100))
    else:
        test_flaring_manual.append(manual)
        if i != 7:
            test_flaring_manual.append(np.zeros(100))
train_flaring_manual = np.concatenate(train_flaring_manual)
test_flaring_manual = np.concatenate(test_flaring_manual)


scaler=StandardScaler()
train=pd.DataFrame(scaler.fit_transform(train_data),columns=train_data.columns)
test=pd.DataFrame(scaler.transform(test_data),columns=test_data.columns)

pca = PCA()
pca.fit(train)

#top 5 pcs
pcs_train = pca.transform(train)[:,:5]
pcs_test = pca.transform(test)[:,:5]


train_states = hmm_model.predict(pcs_train)

test_states = hmm_model.predict(pcs_test)
#%%
#visualize behavior in states
data_means=pd.DataFrame()
for i in range(NUM_STATE):
    mean_state=train[train_states==i].mean()
    data_means['state{}'.format(i)]=mean_state
    
data_means_test=pd.DataFrame()
for i in range(NUM_STATE):
    mean_state=test[test_states==i].mean()
    data_means_test['state{}'.format(i)]=mean_state

TailBeatingDist_train = {}
for i in range(NUM_STATE):
    Tailbeating_instate = np.sum(np.logical_and(train_tailbeat_manual == 1, train_states == i))
    TailBeatingDist_train[i] = Tailbeating_instate

plt.bar(TailBeatingDist_train.keys(), TailBeatingDist_train.values(),color=['black', 'red', 'green', 'blue', 'cyan',"grey"])
plt.xticks(range(NUM_STATE), ["state"+str(i) for i in list(TailBeatingDist_train.keys())])
plt.title("count of tailbeating frames in each state (Train)")

TailBeatingDist_test = {}
for i in range(NUM_STATE):
    Tailbeating_instate = np.sum(np.logical_and(test_tailbeat_manual == 1, test_states == i))
    TailBeatingDist_test[i] = Tailbeating_instate

plt.bar(TailBeatingDist_test.keys(), TailBeatingDist_test.values(),color=['black', 'red', 'green', 'blue', 'cyan',"grey"])
plt.xticks(range(NUM_STATE), ["state"+str(i) for i in list(TailBeatingDist_test.keys())])
plt.title("count of tailbeating frames in each state (Test)")

FlaringDist_train = {}
for i in range(NUM_STATE):
    Flaring_instate = np.sum(np.logical_and(train_flaring_manual == 1, train_states == i))
    FlaringDist_train[i] = Flaring_instate
    
plt.bar(FlaringDist_train.keys(), FlaringDist_train.values(),color=['black', 'red', 'green', 'blue', 'cyan',"grey"])
plt.xticks(range(NUM_STATE), ["state"+str(i) for i in list(FlaringDist_train.keys())])
plt.title("count of flaring frames in each state (Train)")
    
FlaringDist_test = {}
for i in range(NUM_STATE):
    Flaring_instate = np.sum(np.logical_and(test_flaring_manual == 1, test_states == i))
    FlaringDist_test[i] = Flaring_instate
    
plt.bar(FlaringDist_test.keys(), FlaringDist_test.values(),color=['black', 'red', 'green', 'blue', 'cyan',"grey"])
plt.xticks(range(NUM_STATE), ["state"+str(i) for i in list(FlaringDist_test.keys())])
plt.title("count of flaring frames in each state (Test)")
    

data_means.to_csv("data_means.csv")
data_means.to_csv("data_means_test.csv")

#%%
#GLM HMM classification
fish_names = ["IM1_IM2_L", "IM1_IM2_R", "IM2_IM4_L", "IM2_IM4_R",
              "VM1_VM3_L", "VM1_VM3_R","VM3_VM4_L", "VM3_VM4_R"]
AcrossFishPath = sorted(glob("data/AcrossFish/*.csv"))

train_data_clf=pd.DataFrame()
test_data_clf=pd.DataFrame()
train_label=[0] * len(fish_names)
test_label=[0] * len(fish_names)
for i, path in enumerate(fish_names):
    file_path = os.path.join("data", path, "*.csv")
    file = glob(file_path)[0]
    data=pd.read_csv(file)
    freq_feature=get_freq_feature(data)
    data["freq_feature"]=freq_feature
    data=data.iloc[20000:70000,:]
    AcrossFish = pd.read_csv(AcrossFishPath[i//2]).iloc[20000:70000]
    data = pd.concat([data,AcrossFish], axis = 1)
    manual_tailbeat = pd.read_excel(xls, fish_names[i])
    manual_tailbeat = np.array(manual_scoring(manual_tailbeat,200000, crop0 = 90000, crop1 = 140000))
    manual_flare = pd.read_excel(manual_flaring, fish_names[i])
    manual_flare = np.array(manual_scoring(manual_flare,200000, crop0 = 90000, crop1 = 140000))
    manual = manual_tailbeat
    manual[manual_flare == 1] = 2
    if "R" in file: #flip Right images to Left
        data['X_Position']=500-data['X_Position']
        data["orientation"]=180-data["orientation"]
        data["orientation_from_contour"]=180-data["orientation_from_contour"]
    train_data_clf = train_data_clf.append(data.iloc[:40000,:])
    test_data_clf = test_data_clf.append(data.iloc[40000:,:])
    
    if i % 2 == 0:
        train_label[i + 1] = manual[:40000]
        test_label[i + 1] = manual[40000:]
    else:
        train_label[i - 1] = manual[:40000]
        test_label[i - 1] = manual[40000:]
            
        

train_label=np.concatenate(train_label)
test_label=np.concatenate(test_label)

train_data_clf.interpolate(method="linear",inplace=True)
test_data_clf.interpolate(method="linear",inplace=True)

train_data_clf["diff_tail_dev"]=np.diff(train_data_clf.Tail_Dev,prepend=train_data_clf.Tail_Dev.iloc[0])
test_data_clf["diff_tail_dev"]=np.diff(test_data_clf.Tail_Dev,prepend=test_data_clf.Tail_Dev.iloc[0])



#split train and test data, compute tail frequency
train_data=pd.DataFrame()
for i, path in enumerate(fish_names):
    file_path = os.path.join("data", path, "*.csv")
    file = glob(file_path)[0]
    data=pd.read_csv(file)
    freq_feature=get_freq_feature(data)
    data["freq_feature"]=freq_feature
    data=data.iloc[20000:70000,:]
    if "R" in file: #flip Right images to Left
        data['X_Position']=500-data['X_Position']
        data["orientation"]=180-data["orientation"]
        data["orientation_from_contour"]=180-data["orientation_from_contour"]
        #create np.nan filled buffer if neccessary
    train_data = train_data.append(data)
    if i!=7:#last data in train
        buffer=np.full((100,len(feature_columns)),np.nan)
        train_data=train_data.append(pd.DataFrame(buffer,columns=feature_columns))

train_data.interpolate(method="linear",inplace=True)
train_data["diff_tail_dev"]=np.diff(train_data.Tail_Dev,prepend=train_data.Tail_Dev.iloc[0])

scaler=StandardScaler()
train=pd.DataFrame(scaler.fit_transform(train_data),columns=train_data.columns)

pca = PCA()
pca.fit(train)

#top 5 pcs
pcs_train = pca.transform(train)[:,:5]

gaussianHMM=hmm.GaussianHMM(n_components=NUM_STATE,covariance_type="full").fit(pcs_train)
labels=gaussianHMM.predict(pcs_train)

train_states = []
test_states = []
for i in range(8):
    if i % 2 == 0:
        train_states.append(labels[50100 * (i + 1) : 50100 * (i + 1) + 40000])
        test_states.append(labels[50100 * (i + 1) + 40000 : 50100 * (i + 1) + 50000])
    else:
        train_states.append(labels[50100 * (i - 1) : 50100 * (i - 1) + 40000])
        test_states.append(labels[50100 * (i - 1) + 40000 : 50100 * (i - 1) + 50000])
train_states = np.concatenate(train_states)
test_states = np.concatenate(test_states)

enc = OneHotEncoder(handle_unknown='ignore').fit(train_states[:,None])

scaler=StandardScaler()
train_clf=pd.DataFrame(scaler.fit_transform(train_data_clf),columns=train_data_clf.columns)
test_clf = pd.DataFrame(scaler.transform(test_data_clf),columns=test_data_clf.columns)

pca_clf = PCA()
pca_clf.fit(train_clf)



#biplot?
pcs_train_clf = pca_clf.transform(train_clf)[:,:6]
pcs_test_clf = pca_clf.transform(test_clf)[:,:6]

pcs_train_clf_wstates = np.concatenate([pcs_train_clf, enc.transform(train_states[:,None]).toarray()], axis = 1)
pcs_test_clf_wstates = np.concatenate([pcs_test_clf, enc.transform(test_states[:,None]).toarray()], axis = 1)

clf_raw = LogisticRegression(penalty = "l1", class_weight = "balanced", solver = "saga").fit(pcs_train_clf, train_label)
predicted_labels = clf_raw.predict(pcs_test_clf)

cm = confusion_matrix(test_label, predicted_labels)
sn.heatmap(cm, annot = True, xticklabels= ["Other", "TB", "Flare"],
           yticklabels = ["Other", "TB", "Flare"] )
plt.title("Confusion matrix for classification without states")

precision = np.array([cm[0, 0]/np.sum(cm, axis = 0)[0], cm[1,1]/np.sum(cm, axis = 0)[1], cm[2,2]/np.sum(cm, axis = 0)[2]])
recall = np.array([cm[0, 0]/np.sum(cm, axis = 1)[0], cm[1,1]/np.sum(cm, axis = 1)[1], cm[2,2]/np.sum(cm, axis = 1)[2]])
f1 = 2*precision * recall/(precision + recall)

sn.heatmap(np.concatenate([precision[:, None], recall[:, None], f1[:, None]], axis = 1), annot = True, xticklabels = ["Precision", "Recall", "f1"],
           yticklabels = ["Other", "TB", "Flare"])
plt.title("Metrics for classification without states")

#with states
clf_pose = LogisticRegression(penalty = "l1", class_weight = "balanced", solver = "saga").fit(pcs_train_clf_wstates, train_label)
predicted_labels = clf_pose.predict(pcs_test_clf_wstates)

cm = confusion_matrix(test_label, predicted_labels)
sn.heatmap(cm, annot = True, xticklabels= ["Other", "TB", "Flare"],
           yticklabels = ["Other", "TB", "Flare"] )
plt.title("Confusion matrix for classification with states in target fish")

precision = np.array([cm[0, 0]/np.sum(cm, axis = 0)[0], cm[1,1]/np.sum(cm, axis = 0)[1], cm[2,2]/np.sum(cm, axis = 0)[2]])
recall = np.array([cm[0, 0]/np.sum(cm, axis = 1)[0], cm[1,1]/np.sum(cm, axis = 1)[1], cm[2,2]/np.sum(cm, axis = 1)[2]])
f1 = 2*precision * recall/(precision + recall)

sn.heatmap(np.concatenate([precision[:, None], recall[:, None], f1[:, None]], axis = 1), annot = True, xticklabels = ["Precision", "Recall", "f1"],
           yticklabels = ["Other", "TB", "Flare"])
plt.title("Metrics for classification with states in target fish")
