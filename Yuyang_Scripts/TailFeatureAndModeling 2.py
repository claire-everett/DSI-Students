#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:24:18 2020

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
import pywt
from moviepy.video.fx.all import crop,mirror_y
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
from contour_utils import compute_dist,find_centroid,runLengthEncoding
from functions import manual_scoring,find_permutation 
from scipy.signal import argrelextrema,find_peaks
import functools
from glob import glob
from functions import auto_scoring_get_opdeg,find_permutation,permute
IMAGELENGTH=500
fps=40

'''
def window_function(x,minimal=1.5,maximal=5):
    return np.where(np.logical_and(x>minimal,x<maximal),x,0)


def amplitude_scaler(x,f):
    x=x[np.logical_and(f>2,f<10)]
    max_amplitude=np.max(x)
    return float(min(1,max_amplitude/4)**2)

def get_freq_feature(data):
   tail_angle=np.array(data.Tail_Angle)
   tail_angle=np.array(pd.Series(tail_angle).fillna(method="ffill"))
   f, t, Zxx = signal.stft(tail_angle, fs=40,nperseg=70,window = "hann", noverlap=69,boundary=None)
   weighted_freqs_stft=np.matmul(window_function(f,minimal=2,maximal=10).reshape(1,36),np.abs(Zxx)/np.sum(np.abs(Zxx[2:,:]),axis=0)[None,:])
   #flatten 2D array to 1D
   weighted_freqs_stft=weighted_freqs_stft.reshape(weighted_freqs_stft.shape[1])
   amplitude_scale_factor= np.apply_along_axis(amplitude_scaler,0,np.abs(Zxx),f)
   # add a weight factor to local signals that have a small amplitude in 1.5-5HZ so that they get lower scores
   weighted_freqs_stft*=amplitude_scale_factor

   freq_feature=np.concatenate([np.repeat(np.nan,35),weighted_freqs_stft,np.repeat(np.nan,34)])
   
   return freq_feature
'''
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
            
train_data.reset_index(inplace=True)
test_data.reset_index(inplace=True)

train_tail_manual=np.concatenate(train_tail_manual)
test_tail_manual=np.concatenate(test_tail_manual)

train_data.interpolate(method="linear",inplace=True)
test_data.interpolate(method="linear",inplace=True)

train_data["diff_tail_dev"]=np.diff(train_data.Tail_Dev,prepend=train_data.Tail_Dev.iloc[0])
test_data["diff_tail_dev"]=np.diff(test_data.Tail_Dev,prepend=test_data.Tail_Dev.iloc[0])


s=["operculum","orientation_from_contour","X_Position","freq_feature"]

'''
must_include=["operculum","orientation_from_contour","freq_feature"]
debatable=["movement_speed","Tail_Angle","Tail_Dev","diff_tail_dev","curviness","X_Position"]

def all_possible_features(x):
    
    #select all possible subsets for a given set
    
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
    '''
scaler=StandardScaler()
train=pd.DataFrame(scaler.fit_transform(train_data),columns=train_data.columns)
test=pd.DataFrame(scaler.transform(test_data),columns=test_data.columns)

pca = PCA()
pca.fit(train)

#top 5 pcs
pcs_train = pca.transform(train)[:,:5]
pcs_test = pca.transform(test)[:,:5]
'''
#for each feature, run HMM 2 times for each num of cluster

best_result=[]
best_result_test=[]
for s in tqdm(all_possible_sets):
    temp_data=train[s]
    temp_data_test=test[s]
    feature_score={}
    feature_score_test={}
    l=len(s)
    for i in range(4,10):
        meta_scores=[]
        meta_scores_test=[]
        #run HMM 2 times for each n_component
        for j in range(2):
            gaussianHMM=hmm.GaussianHMM(n_components=i,covariance_type="full").fit(temp_data)
            scores=[]
            labels=gaussianHMM.predict(temp_data)
            for l in range(i):
                projected_labels=np.where(labels==l,1,0)
                scores.append(f1_score(train_tail_manual,projected_labels))
            state=np.argmax(scores)
            meta_scores.append(scores[state])
            prejected_labels_test=np.where(gaussianHMM.predict(temp_data_test)==state,1,0)
            meta_scores_test.append(f1_score(test_tail_manual,prejected_labels_test))
        #append avg score for each num_cluster
        avg=np.mean(np.array(meta_scores))
        feature_score[i]=avg
        feature_score_test[i]=np.mean(meta_scores_test)
    #append best result for each set of features
    key=max(feature_score,key=feature_score.get)
    best_result.append([s,key,feature_score[key],feature_score_test[key]])
    key=max(feature_score_test,key=feature_score_test.get)
    best_result_test.append([s,key,feature_score[key],feature_score_test[key]])
result=sorted(best_result,key=lambda x:x[2],reverse=True)
result_test=sorted(best_result_test,key=lambda x:x[2],reverse=True)

with open("feature_scores_updated", 'wb') as f: 
    pickle.dump(result, f)  
'''
'''
scores=[]
for l in range(6):
    projected_labels=np.where(labels==l,1,0)
    scores.append(f1_score(train_tail_manual,projected_labels))
  '''  
'''
#decision tree on a single feature does not really shows a lot of things.
from sklearn.tree import DecisionTreeClassifier, plot_tree
x=np.array(train_data.freq_feature.iloc[:50000]).reshape((50000,1))
clf = DecisionTreeClassifier(max_leaf_nodes=3,min_samples_leaf=2000)
clf.fit(x,Manual1)
plot_tree(clf)
'''

#kde of freq_feature
'''
import seaborn as sn
sn.kdeplot(train_data.freq_feature)

np.mean(train_data.freq_feature<=2)
'''
#train_s=train[s]
#test_s=test[s]

num_states=6
gaussianHMM=hmm.GaussianHMM(n_components=num_states,covariance_type="full").fit(pcs_train)
labels=gaussianHMM.predict(pcs_train)

test_labels=gaussianHMM.predict(pcs_test)

scores = []
for l in range(i):
    projected_labels=np.where(labels==l,1,0)
    scores.append(f1_score(train_tail_manual,projected_labels))

test_scores = []
for l in range(i):
    projected_labels=np.where(test_labels==l,1,0)
    test_scores.append(f1_score(test_tail_manual,projected_labels))



data_means=pd.DataFrame()
for i in range(num_states):
    mean_state=train[labels==i].mean()
    data_means['state{}'.format(i)]=mean_state
    
import contour_utils
np.sum(labels == 5)
value, length = contour_utils.runLengthEncoding(labels == 5)
np.mean(length[value])

data_means_test=pd.DataFrame()
for i in range(num_states):
    mean_state=test[test_labels==i].mean()
    data_means_test['state{}'.format(i)]=mean_state
    
with open("models/GaussianHMM_6states.pkl","wb") as fp:
    pickle.dump(gaussianHMM,fp)

#%% discrete hmm
def discretize(data):
    new_data=data.copy()
    oper=new_data.operculum
    oper[data.operculum>63]="Oper_open"
    oper[data.operculum<=63]="Oper_close"
    
    curviness=new_data.curviness
    curviness[data.curviness>0.6]="curved"
    curviness[data.curviness<=0.6]="flat"
    
    orientation=new_data.orientation_from_contour
    orientation[data.orientation_from_contour<30]="ori:0"
    orientation[data.orientation_from_contour>=30]="ori:1"
    orientation[data.orientation_from_contour>=70]="ori:2"
    
    freq_feature=new_data.freq_feature
    freq_feature[data.freq_feature>2.5]="high_freq"
    freq_feature[data.freq_feature<=2.5]="mid_freq"
    freq_feature[data.freq_feature<=2]="low_freq"
    '''
    orientation=new_data.orientation
    orientation[data.orientation<30]="ori:0"
    orientation[data.orientation>=30]="ori:1"
    orientation[data.orientation>=70]="ori:2"
    '''
    new_data["operculum"]=oper
    new_data["curviness"]=curviness
    new_data["orientation_from_contour"]=orientation
    new_data["freq_feature"]=freq_feature
    '''
    new_data["orientation"]=orientation
    '''
    return new_data
train_s=train_data[s]
test_s=test_data[s]
train_s=discretize(train_s)
test_s=discretize(test_s)

train_s["label"]=train_s.operculum+","+train_s.orientation_from_contour+","+train_s.curviness+","+train_s.freq_feature
test_s["label"]=test_s.operculum+","+test_s.orientation_from_contour+","+test_s.curviness+","+test_s.freq_feature

'''
train_s["label"]=train_s.operculum+","+train_s.orientation+","+train_s.curviness
test_s["label"]=test_s.operculum+","+test_s.orientation+","+test_s.curviness
'''
popular_labels=train_s.label.value_counts()[train_s.label.value_counts()>1500].index
train_s["label"] = np.where(train_s["label"].isin(popular_labels), train_s["label"], 'OTHER')
test_s["label"] = np.where(test_s["label"].isin(popular_labels), test_s["label"], 'OTHER')

#encode the discretized feature to 1 label column
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoded_labels=encoder.fit_transform(train_s.label).reshape((len(train_s),1))
corresponding_types=encoder.inverse_transform(range(np.unique(encoded_labels)[-1]+1))


n_states=4
mnHMM=hmm.MultinomialHMM(n_components=n_states).fit(encoded_labels)

#the emission prob
cols=list(range(n_states))
cols=["state"+str(cols[i]) for i in cols]
emission=pd.DataFrame(mnHMM.emissionprob_.T,columns=cols)
emission.index=corresponding_types

#the label
labels=mnHMM.predict(encoded_labels)

#permute the labels to the previous hmm's label
perms=find_permutation(labels,prev_labels)
labels=permute(labels,perms)



#labels on test set
encoded_labels_test=encoder.transform(test_s.label).reshape((len(test_s),1))

test_labels=mnHMM.predict(encoded_labels_test)
test_labels=permute(test_labels,perms)


with open("models/multinomialHMM_4states.pkl","wb") as fp:
    pickle.dump(mnHMM,fp)

#count of different combinations of features
for i in range(len(corresponding_types)):
    count=np.sum((train_s.label==corresponding_types[i]))
    print("class {} :".format(corresponding_types[i]))
    print("count = {}".format(count))


#make video

'''
train:
IM1_IM2_L:0-50000
IM1_IM2_R:50100-100100
IM2_IM4_L:100200-150200
IM2_IM4_R:150300-200300
VM1_VM3_L:200400-250400
VM1_VM3_R:250500-300500

test:
VM3_VM4_L:0-50000
VM3_VM4_R:50100-101000
'''
'''
def visualize_result(path,out,data,discretized_data,labels,start_index,data_start_index,feature_list):
    '''
'''

    Parameters
    ----------
    path : path of the video file to read
    out: video name to write 
    data: the combined dataframe
    discretized_data: the discretized_data
    labels:labels predicted by HMM
    start_index : the starting index in the combined dataframe
    data_start_index : the index of the one particular fish in the combined data, like 50100 for IM1_IM2_R
    feature_list : list of features to be written on video.

    Returns
    -------
    None.
'''
'''
    cap = cv2.VideoCapture(path) 
    index=90000+start_index-data_start_index
    colormap=[(255,0,0), (51,51,255), (51,255,51),(51,255,255),(0,0,0),(255,128,0),(255,255,255),(255,255,0),(204,0,204),(160,160,160)]
    cap.set(1,index)
    result = cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGELENGTH,IMAGELENGTH))
    iterator=tqdm(range(10000))
    for i in iterator: 
        # Capture frames in the video 
        ret, frame = cap.read() 
        if ret!=True:
            break
        label=labels[start_index+i]
        #value of features
        color=colormap[label]
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
    
        #write curviness
        for j,feature in enumerate(feature_list):
            feature_name=feature if feature!="orientation_from_contour" else "orientation"
            frame=cv2.putText(frame,  
                feature_name+" = {}, type= {}".format(round(data[feature].iloc[start+i],2),discretized_data[feature].iloc[start+i]),  
                (0, 450-j*20),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
        frame=cv2.putText(frame,  
                "{} frame".format(index+i),  
                (0, 470),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
        result.write(frame)
    result.release()
'''
def visualize_result(path,out,data,Manual,labels,start_index,data_start_index,feature_list):
    '''

    Parameters
    ----------
    path : path of the video file to read
    out: video name to write 
    data: the combined dataframe
    Manual: concatenated manual score from excel
    labels:labels predicted by HMM
    start_index : the starting index in the combined dataframe
    data_start_index : the index of the one particular fish in the combined data, like 50100 for IM1_IM2_R
    feature_list : list of features to be written on video.

    Returns
    -------
    None.

    '''
    cap = cv2.VideoCapture(path) 
    index=90000+start_index-data_start_index
    colormap=[(255,0,0), (51,51,255), (51,255,51),(51,255,255),(0,0,0),(255,128,0),(255,255,255),(255,255,0),(204,0,204),(160,160,160)]
    cap.set(1,index)
    result = cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGELENGTH,IMAGELENGTH))
    iterator=tqdm(range(10000))
    for i in iterator: 
        # Capture frames in the video 
        ret, frame = cap.read() 
        if ret!=True:
            break
        label=labels[start_index+i]
        manual=Manual[start_index+i]
        manual_color= (0,0,0) if manual==1 else (255,255,255)
        #value of features
        color=colormap[label]
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
    
        #write features
        for j,feature in enumerate(feature_list):
            feature_name=feature if feature!="orientation_from_contour" else "orientation"
            frame=cv2.putText(frame,  
                feature_name+" = {}".format(round(data[feature].iloc[start+i],2)),  
                (0, 450-j*20),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
        frame=cv2.putText(frame,  
                "{} frame".format(index+i),  
                (0, 470),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
        result.write(frame)
    result.release()    
'''
start=np.random.randint(150300,190300,1)[0]

visualize_result("TailBeatingExamples/Copy of IM2_IM4_5.1.2_R.mp4",'videos/hmm_6_states_test_IM2_IM4_R.mp4',
                 train_data,train,labels,start,150300,[""])
'''
start=np.random.randint(150300,190300,1)[0]

visualize_result("TailBeatingExamples/Copy of IM2_IM4_5.1.2_R.mp4",'videos/hmm_6_states_test_IM2_IM4_R.mp4',
                 train_data,train_tail_manual,labels,start,150300,[])

start=np.random.randint(50100,90100,1)[0]

visualize_result("TailBeatingExamples/Copy of VM3_VM4_5.1.1_R.mp4",'videos/hmm_6_states_test_VM3_VM4_R.mp4',
                 test_data,test_tail_manual,test_labels,start,50100,[])
