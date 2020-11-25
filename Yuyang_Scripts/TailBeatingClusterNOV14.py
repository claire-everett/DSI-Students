#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:22:34 2020

@author: ryan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sn
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
from scipy.signal import find_peaks
IMAGELENGTH=500
fps=40


data_auto_scored = pd.read_csv("data/IM1_IM2_2.1.1_L_data_auto_scored.csv")
data_manual = pd.read_excel("Claire_score_tables/IM1_IM2_L_Manual_score.xlsx")

starttime = 90000
endtime = 140000
Manual1 = np.array(manual_scoring(data_manual,200000, crop0 = starttime, crop1 = endtime))

tail_angle=np.array(data_auto_scored.Tail_Angle)
f, t, Zxx = signal.stft(tail_angle, fs=40,nperseg=70,window = "hann", noverlap=69,boundary=None)


tail_angle=np.array(pd.Series(tail_angle).fillna(method="ffill"))

#a visualization on the signal and its corresponding fourier transformation
def Angle_make_frame(t,angle,Zxx,step,start_frame,to_array=True,find_local_max=False):
    time=int(t*40)
    time=time+start_frame
    start=max(0,time-200)
    stop=min(79999,time+200)
    feature=freq_feature[time]
    fig,ax=plt.subplots(2,1)
    if time-step>=start and time+step<=stop:
        ax[0].plot(range(time-step,time+step),angle[time-step:time+step],color="r")
        ax[0].plot(range(start,time-step),angle[start:time-step],color="b")
        ax[0].plot(range(time+step,stop),angle[time+step:stop],color="b")
        index=time-step
    elif time-step<start:
        ax[0].plot(range(start,time+step),angle[start:time+step],color="r")
        ax[0].plot(range(time+step,stop),angle[time+step:stop],color="b")
        index=np.nan
    else:
        ax[0].plot(range(time-step,stop),angle[time+step:stop],color="r")
        ax[0].plot(range(start,time-step),angle[start:time-step],color="b")
        index=np.nan
    ax[0].plot(time,angle[time],"ro")
    ax[0].set_ylim(0,190)
    ax[0].set_ylabel("degree")
    ax[0].text(start-10, 10,"{} frame".format(time+70000),bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax[0].text(start-10, -20,"freq_feautre={}".format(round(feature,2)),bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax[0].set_xticks([])
    if ~np.isnan(index):
        if find_local_max:
            local_max=find_peaks(np.abs(Zxx)[2:,index]/np.max(np.abs(Zxx[2:,index])),prominence=0.15)[0]
            ax[1].scatter(f[2:][local_max],np.abs(Zxx)[2:,index][local_max],c="red")
        ax[1].plot(f[2:],np.abs(Zxx)[2:,index])
        ax[1].set_title("FFT on segment")
        ax[1].set_ylabel("amplitude")
        #ax[1].set_yticks([])
    if to_array:
        out=mplfig_to_npimage(fig)
        plt.close(fig)
        return out
    else:
        plt.show()
import functools
start=20000
end=22000
duration=(end-start)/40

make_frame=functools.partial(Angle_make_frame,angle=tail_angle,Zxx=Zxx,step=35,start_frame=start,to_array=True,find_local_max=True)


animation = VideoClip(make_frame, duration = duration)
animation.write_videofile("videos/IM1_IM2_L_signal_test.mp4", fps=40)

video = VideoFileClip("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L_timestamp.mp4")
video=video.margin(10)
video=video.subclip((70000+start)/fps,(70000+end)/fps)
video_signal = VideoFileClip("videos/IM1_IM2_L_signal_test.mp4")

final_clip = clips_array([[video, video_signal]])
final_clip.write_videofile("videos/IM1_IM2_wSignal_test.mp4")


def window_function(x,minimal=1.5,maximal=5):
    return np.where(np.logical_and(x>minimal,x<maximal),x,0)


weighted_freqs_stft=np.matmul(window_function(f,minimal=2,maximal=10).reshape(1,36),np.abs(Zxx)/np.sum(np.abs(Zxx[2:,:]),axis=0)[None,:]).reshape(79931)


def amplitude_scaler(x,f):
    x=x[np.logical_and(f>2,f<10)]
    max_amplitude=np.max(x)
    return float(min(1,max_amplitude/4)**2)
amplitude_scale_factor= np.apply_along_axis(amplitude_scaler,0,np.abs(Zxx),f)
# add a weight factor to local signals that have a small amplitude in 1.5-5HZ so that they get lower scores
weighted_freqs_stft*=amplitude_scale_factor

freq_feature=np.concatenate([np.repeat(np.nan,35),weighted_freqs_stft,np.repeat(np.nan,34)])

data_auto_scored['tail_freq_score']=freq_feature

with open("data/IM1_IM2_2.1.1_L_70000_150000_curve_scores_NOV12", "rb") as fp:   
    curve_scores = pickle.load(fp)  
    
del data_auto_scored['Unnamed: 0']

data=data_auto_scored.iloc[20000:70000,:]
scaler=StandardScaler()
data=pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
data=data.fillna(method="ffill")

correlation=data.corr()

s=["operculum","orientation","movement_speed","X_Position","tail_freq_score","curviness"]
'''
s=["operculum","orientation","movement_speed","X_Position","tail_freq_score"]
s=["operculum","orientation","movement_speed","tail_freq_score"]
'''
temp_data=data[s]
from sklearn.decomposition import PCA
pca=PCA().fit(temp_data)
pcs=pca.transform(temp_data)[:,:3]

components=pca.components_
#pc0: -oper +orientation, - X_Position
#pc1: +mov_speed, +tail_freq, +curviness
#pc2: +oper, +mov_speed, -X_position
n_states=7
gaussianHMM=hmm.GaussianHMM(n_components=n_states,covariance_type="full").fit(pcs)
scores=[]
labels=gaussianHMM.predict(pcs)
for l in range(n_states):
    projected_labels=np.where(labels==l,1,0)
    scores.append(f1_score(Manual1,projected_labels))
print(scores)
means=pd.DataFrame(gaussianHMM.means_)

temp_data["label"]=labels
data_means=pd.DataFrame()
for i in range(n_states):
    mean_state=temp_data[temp_data.label==i].drop("label",axis=1).mean()
    data_means['state{}'.format(i)]=mean_state
    
data_var=pd.DataFrame()
for i in range(n_states):
    var_state=temp_data[temp_data.label==i].drop("label",axis=1).var()
    data_var['state{}'.format(i)]=var_state

#sum of duration
for i in range(n_states):
    print(len(labels[labels==i]))

#average duration for each state
value,length=runLengthEncoding(labels)
rle_df=pd.DataFrame({"value":value,"length":length})
for i in range(n_states):
    print(rle_df[rle_df.value==i].length.mean())
import cv2 
fps=40
IMAGELENGTH=500
# path 
path = "TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"
max_freq_feature=np.nanmax(freq_feature)
cap = cv2.VideoCapture(path) 
index=90000
colormap=[(255,0,0), (51,51,255), (51,255,51),(51,255,255),(0,0,0),(255,128,0),(255,255,255),(255,255,0),(204,0,204),(160,160,160)]
cap.set(1,index)
result = cv2.VideoWriter('videos/hmm_5_states_test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGELENGTH,IMAGELENGTH))
iterator=tqdm(range(10000))
for i in iterator: 
    # Capture frames in the video 
    ret, frame = cap.read() 
    if ret!=True:
        break
    label=labels[i]
    feature=freq_feature[i+20000]
    manual=Manual1[i]
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

with open("models/HMM_7states_NOV15.pkl","wb") as fp:
    pickle.dump(gaussianHMM,fp)
    
fig = plt.figure()
ax = plt.axes(projection='3d')
scatter=ax.scatter3D(pcs[:,0],pcs[:,1],pcs[:,2],c=labels,s=0.1)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
plt.title("hidden state distribution on the 3 pcs")

'''
7states:
state0: tail beating
state1: "static" flaring: head straight, not moving much
state2 :turn +flare: it is still likely flaring, but also making a big turn in the meantime
state3: facing N/S, with a little tail beating, most likely to be the state before or after tail beating
state4: moving, i.e. high speed
state5: looks like transition between states, happens when like fish is flaring in a tilted direction, or other transition between states
state6: idling, not very often
'''   

#%% 
# the same process for the right fish, not much different with the upper part


data_auto_scored_R = pd.read_csv("data/IM1_IM2_2.1.1_R_data_auto_scored.csv")

tail_angle_R=np.array(data_auto_scored_R.Tail_Angle)
tail_angle_R=np.array(pd.Series(tail_angle_R).fillna(method="ffill"))

f_R, t_R, Zxx_R = signal.stft(tail_angle_R, fs=40,nperseg=70,window = "hann", noverlap=69,boundary=None)

weighted_freqs_stft_R=np.matmul(window_function(f_R).reshape(1,36),np.abs(Zxx_R)/np.sum(np.abs(Zxx_R[2:,:]),axis=0)[None,:]).reshape(79931)

amplitude_scale_factor_R= np.apply_along_axis(func1d=amplitude_scaler,axis=0,arr=np.abs(Zxx_R),f=f_R)
# add a weight factor to local signals that have a small amplitude in 1.5-5HZ so that they get lower scores
weighted_freqs_stft_R*=amplitude_scale_factor_R

freq_feature_R=np.concatenate([np.repeat(np.nan,35),weighted_freqs_stft_R,np.repeat(np.nan,34)])

data_auto_scored_R['tail_freq_score']=freq_feature_R

data_auto_scored_R["orientation"]=180-data_auto_scored_R["orientation"]
data_auto_scored_R["X_Position"]=500-data_auto_scored_R["X_Position"]

del data_auto_scored_R['Unnamed: 0']

data_R=data_auto_scored_R.iloc[20000:70000,:]
scaler_R=StandardScaler()
data_R=pd.DataFrame(scaler_R.fit_transform(data_R),columns=data_R.columns)
data_R=data_R.fillna(method="ffill")

temp_data_R=data_R[s]
'''
from sklearn.decomposition import PCA
pca_R=PCA().fit(temp_data_R)
'''
pcs_R=pca.transform(temp_data_R)[:,:3]


#pc0: -oper +orientation, - X_Position
#pc1: +mov_speed, +tail_freq, +curviness
#pc2: +oper, +mov_speed, -X_position
n_states=7
gaussianHMM_R=hmm.GaussianHMM(n_components=n_states,covariance_type="full").fit(pcs_R)
labels_R=gaussianHMM_R.predict(pcs_R)


temp_data_R["label"]=labels_R
#test=compute_state_overlap(labels_R,labels,temp_data_R,temp_data)
perms=find_permutation(labels_R,labels,temp_data_R,temp_data)

temp_data_R["label"]=labels_R
data_means_R=pd.DataFrame()
for i in range(n_states):
    mean_state=temp_data_R[temp_data_R.label==i].drop("label",axis=1).mean()
    data_means_R['state{}'.format(i)]=mean_state
    
data_var_R=pd.DataFrame()
for i in range(n_states):
    var_state=temp_data_R[temp_data_R.label==i].drop("label",axis=1).var()
    data_var_R['state{}'.format(i)]=var_state

#sum of duration
for i in range(n_states):
    print(len(labels[labels==i]))

#average duration for each state
value,length=runLengthEncoding(labels_R)
rle_df=pd.DataFrame({"value":value,"length":length})
for i in range(n_states):
    print(rle_df[rle_df.value==i].length.mean())
    
# path 
path = "TailBeatingExamples/Copy of IM1_IM2_2.1.1_R.mp4"
max_freq_feature=np.nanmax(freq_feature_R)
cap = cv2.VideoCapture(path) 
index=90000
colormap=[(255,0,0), (51,51,255), (51,255,51),(51,255,255),(0,0,0),(255,128,0),(255,255,255),(255,255,0),(204,0,204),(160,160,160)]
cap.set(1,index)
result = cv2.VideoWriter('videos/hmm_7_states_Rtest.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGELENGTH,IMAGELENGTH))
iterator=tqdm(range(10000))
for i in iterator: 
    # Capture frames in the video 
    ret, frame = cap.read() 
    if ret!=True:
        break
    label=labels_R[i]
    feature=freq_feature_R[i+20000]
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

with open("data/IM1_IM2_2.1.1_R_70000_150000_head_index", "rb") as fp:   
    head_indexs = pickle.load(fp) 
with open("data/IM1_IM2_2.1.1_R_70000_150000_tail_index", "rb") as fp:   
    tail_indexs = pickle.load(fp) 

def make_frame(t,start_frame=0,to_array=False):
    time=int(t*40)+start_frame
    curve_score=curve_scores_R[time]
    head_index=head_indexs[time]
    tail_index=tail_indexs[time]
#visualize the contour with the curveness as heatmap, and head and tail on it
    fig=plt.figure()
    contour=curve_score[:,:2]      
    plt.scatter(curve_score[:,0], curve_score[:,1], c=curve_score[:,2], s=3,cmap="YlGnBu", vmin=0, vmax=1)
    plt.plot(np.float64(contour[tail_index,0]),np.float64(contour[tail_index,1]),"ro",markersize=5,color="red")
    plt.plot(np.float64(contour[head_index,0]),np.float64(contour[head_index,1]),"ro",markersize=5,color="purple")
    plt.xlim(0,500)
    plt.ylim(0,500)
    plt.colorbar()
    plt.text(25,25,"frame {}".format(time))
    if to_array:
        out=mplfig_to_npimage(fig)
        plt.close(fig)
        return out
    else:
        plt.show()
    
make_frame_R=functools.partial(make_frame,start_frame=20000,to_array=True)


animation = VideoClip(make_frame_R, duration = 60)
animation.write_videofile("videos/test.mp4", fps=40)


max_operculum=np.max(data_auto_scored_R.operculum)
max_speed=np.max(data_auto_scored_R.movement_speed[20000:70000])
max_freq_feature=np.max(data_auto_scored_R.tail_freq_score[20000:70000])
max_curviness=np.max(data_auto_scored_R.curviness[20000:70000])
max_orientation=np.max(data_auto_scored_R.orientation[20000:70000])
path = "TailBeatingExamples/Copy of IM1_IM2_2.1.1_R_timestamp.mp4"
cap = cv2.VideoCapture(path) 
index=90000
cap.set(1,index)
IMAGELENGTH=500
result = cv2.VideoWriter('videos/features_visualization_IM1_IM2_R.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGELENGTH,IMAGELENGTH))
font = cv2.FONT_HERSHEY_SIMPLEX 
iterator=range(10000)
for i in iterator: 
    # Capture frames in the video 
    ret, frame = cap.read() 
    if ret!=True:
        break
    time=i+20000
    #feature value
    operculum=data_auto_scored_R.operculum.iloc[time]
    speed=data_auto_scored_R.movement_speed.iloc[time]
    cur_freq_feature=data_auto_scored_R.tail_freq_score.iloc[time]
    body_curviness=data_auto_scored_R.curviness.iloc[time]
    orientation=data_auto_scored_R.orientation.iloc[time]
    #add bars
    frame=cv2.rectangle(frame,(300,225),(300+int(orientation/max_orientation*150),238),(0,0,0),-1)
    frame=cv2.putText(frame,  
                "orientation = {}".format(round(orientation,2)),  
                (300, 250),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    frame=cv2.rectangle(frame,(300,275),(300+int(body_curviness/max_curviness*150),288),(0,0,0),-1)
    frame=cv2.putText(frame,  
                "curviness = {}".format(round(body_curviness,2)),  
                (300, 300),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    frame=cv2.rectangle(frame,(300,325),(300+int(speed/max_speed*150),338),(0,0,0),-1)
    frame=cv2.putText(frame,  
                "speed = {}".format(round(speed,2)),  
                (300, 350),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    frame=cv2.rectangle(frame,(300,375),(300+int(operculum/max_operculum*150),388),(0,0,0),-1)
    frame=cv2.putText(frame,  
                "Operculum_angle = {}".format(round(operculum,2)),  
                (300, 400),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    frame=cv2.rectangle(frame,(300,425),(300+int(cur_freq_feature/max_freq_feature*150),438),(0,0,0),-1)
    frame=cv2.putText(frame,  
                "frequency feature = {}".format(round(cur_freq_feature,2)),  
                (300, 450),  
                font, 0.5,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4) 
    result.write(frame)
result.release()
#%% revising curviness score, trying to remove pelvic fin, still testing
'''
size=500
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(cv2.drawContours(np.float32(np.full((size,size),255)),[new_contour],0,(0,0,0),cv2.FILLED))
plt.plot(new_contour[first_convex_ind%N,0],new_contour[first_convex_ind%N,1],"ro")
plt.plot(new_contour[next_convex_ind%N,0],new_contour[next_convex_ind%N,1],"ro")
plt.plot(new_contour[(first_convex_ind+int(l/2))%N,0],new_contour[(first_convex_ind+int(l/2))%N,1],"ro")
circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
ax.add_patch(circ)

plt.imshow(cv2.drawContours(np.float32(np.full((500,500),255)),[new_contour],0,(0,0,0),cv2.FILLED))
index=133
plt.plot(new_contour[index,0],new_contour[index,1],"ro")
plt.plot(new_contour[27,0],new_contour[27,1],"ro")
plt.plot(new_contour[index+15,0],new_contour[index+15,1],"ro")

with open("data/IM1_IM2_2.1.1_R_70000_150000_curve_scores", "rb") as fp:   
    body_curviness_scores_R=pickle.load(fp)  
new_contours=[]
iterator=tqdm(range(len(body_curviness_scores_R)))
for i in iterator:
    curve_score=curve_scores_R[i]
    new_contour=smoothConvexPart(curve_score,quantile=0.6)
    new_contours.append(new_contour)
    
new_curve_scores=[] 
iterator=tqdm(range(len(new_contours)))
for i in iterator:
    contour=new_contours[i]
    contour=np.array(contour.reshape((len(contour),1,2)),dtype=np.int32)
    new_curve_score=compute_cos_fullbody(contour)
    new_curve_scores.append(new_curve_score)

with open("data/IM1_IM2_2.1.1_R_70000_150000_curve_scores_refined", "wb") as fp:   
    pickle.dump(new_curve_scores,fp)     

def make_frame(t,start_frame=0,to_array=False,quantile=0.4):
    time=int(t*40)+start_frame
    curve_score=new_curve_scores[time]
    contour=np.array(curve_score[:,:2].reshape(len(curve_score),1,2),dtype=np.int32)
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    fig=plt.figure() 
    ax = fig.add_subplot(1,1,1)    
    plt.scatter(curve_score[:,0], curve_score[:,1], c=curve_score[:,2], s=3,cmap="YlGnBu", vmin=0, vmax=1)
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.xlim(0,500)
    plt.ylim(0,500)
    plt.colorbar()
    plt.text(25,25,"frame {}".format(time))
    if to_array:
        out=mplfig_to_npimage(fig)
        plt.close(fig)
        return out
    else:
        plt.show()
    
make_frame=functools.partial(make_frame,start_frame=0,to_array=True,quantile=0.4)


animation = VideoClip(make_frame, duration = 50)
animation.write_videofile("videos/test.mp4", fps=40)
'''