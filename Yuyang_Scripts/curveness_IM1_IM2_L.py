#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:15:49 2020

@author: ryan
"""
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
from moviepy.video.fx.all import crop
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
from tqdm import tqdm
from scipy.ndimage import zoom  
import warnings
from contour_utils import compute_cos,plot_result,find_centroid,visualize_steps,find_anchor_rle,visualize_segments
from scipy import interpolate 
from numpy import random
from find_features import features
import seaborn as sn
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import gizeh
import moviepy.video.compositing as mp
#vidcap = cv2.VideoCapture("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4")
#success,image=vidcap.read()

image_length=500
fps=40
videopath="TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"
def find_contour(videopath,length=40*300,interpolate=True,start=0,step=1,pre_filter=None):
    vidcap = cv2.VideoCapture(videopath)
    img_array=[]
    mask_array=[]
    contour_array=[]
    length=int(length/step)
    index=start
    vidcap.set(1,index)
    for i in tqdm(range(length)):
        success,image=vidcap.read()
        if success!=1:
            print("process stops early at {}th iteration".format(i))
            break
        index+=step
        vidcap.set(1,index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        th = cv2.dilate(th, k2, iterations=1)
    #th=cv2.erode(th,None,iterations=1)
        if pre_filter is not None:
            th=pre_filter(th)
        contours, hierarchy = cv2.findContours(np.invert(th.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        flag=0
        area=0
        for cnt in contours:
            new_area=cv2.contourArea(cnt)
            if new_area>area and new_area<10000 :#in case some contour contains almost the whole image, not required if invert the image first
                area=new_area
                fish_contour=cnt
                flag=1
        if flag==0:
            print("no valid fish contour find at index {}".format(i))
            img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
        else:
            img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=np.invert(np.array(cv2.erode(img,k2,iterations=1),np.uint8))
        if interpolate==True:
            img=zoom(img,3)
            img=cv2.medianBlur(img, 21) #since zoom made the img larger
        else:
             img=cv2.medianBlur(img, 7)
        true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        try:
            contour_array.append(true_contour[0])
        except:
            contour_array.append(np.zeros((1,2),dtype=np.float64))
        img_array.append(np.array(cv2.drawContours(img.copy(),true_contour,0,(0,255,0),2),np.uint8))
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1],3),255)),true_contour,0,(0,0,0),cv2.FILLED),np.uint8))
    return img_array,mask_array,contour_array

img_array,mask_array,contour_array=find_contour("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4",length=5000,start=90000,step=1,interpolate=True)

out = cv2.VideoWriter('videos/IM1_IM2_contour_median7_test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(img_array)):
     out.write(img_array[i])
out.release()

samples=np.random.randint(0,4999,5)
for i in samples:
    contour=contour_array[i]
    out,_1,_2=compute_cos(contour,step=30,img_size=1500,min_step=30)
    plot_result(out,contour,img_size=1500)

#dead kernel warning
#take the top60 score at each point
'''
currently I am not really sure I can store all segmentation's score at each point as my kernel keft dying if
I just use seg1s.append(seg1) sort of thing
'''
topCos=np.zeros((0,30))
plots=[]
for contour in tqdm(contour_array):
    out,_,_=compute_cos(contour,step=30,img_size=1800,min_step=30)
    plots.append(plot_result(out,contour,img_size=1800,to_array=True))
    topCos=np.vstack([topCos,np.sort(out.flatten())[::-1][:30]])
    

#there's a small bug here that even after blurring the new  mask still has more than 1 contour, should select the
#largest contour again, it is very rare though
topCos[np.isnan(topCos)]=0
#pca, and take 1st pc 
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pcs=pca.fit_transform(topCos)
  

#simple cluster to put the curveness into 2 different labels
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(pcs)
labels=kmeans.predict(pcs)

#make video
def make_frame(time):
    timeint = int(time*fps)
    return plots[timeint]

def make_frame_label(time):
    timeint = int(time*fps)
    col1 = labels[timeint]
    col1 = col1/4
    #print(col1)
    surface = gizeh.Surface(128,128, bg_color=(1,1,1,.5))
    if col1 == 0:
        circle = gizeh.circle(25, xy = (64,64), fill=(1,1,1))
    if col1 == .25:
         circle = gizeh.circle(25, xy = (64,64), fill=(0,0,0))
    circle.draw(surface)
    return surface.get_npimage()
  
starttime=90000
duration=125
clip1=VideoClip(make_frame,duration=125)
clip2 = VideoClip(make_frame_label, duration = 125)
composite_clip = mp.CompositeVideoClip.CompositeVideoClip([clip1,clip2.set_position((0,0))]) 
composite_clip.write_videofile("videos/curveness_label.mp4", fps = 40)
    
    
    


#%%
'''
below is the classification I tested
tbh it only improved a little bit from ROC and confusion matrix, and tail beating is still kind of confusing to me
'''
path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()

#importing data
def manual_scoring(data_manual,data_auto,crop0 = 0,crop1= -1):
    '''
    A function that takes manually scored data and converts it to a binary array. 
    
    Parameters: 
    data_manual: manual scored data, read in from an excel file
    data_auto: automatically scored data, just used to establish how long the session is. 
    
    Returns: 
    pandas array: binary array of open/closed scoring
    '''
    Manual = pd.DataFrame(0, index=np.arange(len(data_auto)), columns = ['OpOpen'])
    reference = data_manual.index
    
    
    for i in reference:
        Manual[data_manual['Start'][i]:data_manual['Stop'][i]] = 1
    
    print(Manual[data_manual['Start'][i]:data_manual['Stop'][i]]) 
     
    return Manual['OpOpen'][crop0:crop1]

excel_files = "TailManual.xlsx"
file_handle1 = excel_files
data_manual1 = pd.read_excel(file_handle1)
starttime=90000
endtime=140000
Manual1 = manual_scoring(data_manual1, df, crop0 = starttime, crop1 =  endtime)
y=np.array(Manual1)

new_features=features(starttime=starttime,endtime=endtime)
filtered_df=new_features.filter_df(df,add_midpoint=True)
new_features.fit(filtered_df,filter_feature=True,fill_na=True,estimate_na=False)
#get other_features,i.e.:speed,turning_angle,operculum,orientation
other_features,curvatures,diff_curvatures,tangent=new_features.export_df()

#take every 10 frames  
other_features=other_features.iloc[::10,:]
y=y[::10]

curve_score=pcs
diff_curve_score=np.abs(np.append(0,np.diff(curve_score.squeeze())))
other_features.reset_index(drop=True, inplace=True)
possible_features=pd.concat([other_features,pd.Series(curve_score.flatten()),pd.Series(diff_curve_score)],axis=1)
possible_features=possible_features.rename(columns={0:"curve_score",1:"diff_curve_score"})

First_period=possible_features.iloc[:3000,:]
Second_period=possible_features.iloc[3000:4000,:]
y1=y[:3000]
y2=y[3000:4000]


xgbClassifier=XGBClassifier(max_depth=1,scale_pos_weight=9)
xgbClassifier.fit(First_period,y1)
pred_train=xgbClassifier.predict(First_period)
#confusion matrix on train period
sn.heatmap(confusion_matrix(y1,pred_train),annot=True,cmap = "YlGnBu")



pred_test=xgbClassifier.predict(Second_period)
#confusion matrix on test period
sn.heatmap(confusion_matrix(y2,pred_test),annot=True,cmap = "YlGnBu")
pred_prob=xgbClassifier.predict_proba(Second_period)[:,1]
#ROC curve
auc=roc_auc_score(y2,pred_prob)
fpr, tpr, threshold=roc_curve(y2,pred_prob)
plt.title('ROC curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#confusion matrix according to threshold picked from roc curve
th=np.where(fpr<0.2)
pred_test=pred_prob>threshold[th]
sn.heatmap(confusion_matrix(y2,pred_test),annot=True,cmap = "YlGnBu")
#visualization on how pred and true label goes through time
def make_frame(time):
    timeint = int(time*fps)
    col1 = pred_test[int(timeint/10)]
    col1 = col1/4
    #print(col1)
    surface = gizeh.Surface(128,128, bg_color=(1,1,1,.5))
    #fill= rgb/255
    if col1 == 0:
        circle = gizeh.circle(50, xy = (64,64), fill=(1,1,1))
    if col1 == .25:
        circle = gizeh.circle(50, xy = (64,64), fill=(0,0, 0))
    circle.draw(surface)
    return surface.get_npimage()
animation = VideoClip(make_frame, duration = 200)
animation.write_videofile("videos/pred_test_period.mp4", fps=40)

y=np.array(Manual1)
y1=y[:30000]
y2=y[30000:40000]
def make_frame_manual(time):
    timeint = int(time*fps)
    col1 = y2[timeint]
    col1 = col1/4
    #print(col1)
    surface = gizeh.Surface(128,128, bg_color=(1,1,1,.5))
    if col1 == 0:
        circle = gizeh.circle(50, xy = (64,64), fill=(1,1,1))
    if col1 == .25:
         circle = gizeh.circle(50, xy = (64,64), fill=(0,0,0))
    circle.draw(surface)
    return surface.get_npimage()

animation_manual = VideoClip(make_frame_manual, duration = 200)
animation_manual.write_videofile("videos/Manual_tail_test_period.mp4", fps=40)


starttime=120000
duration=200
clip1 = VideoFileClip("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4").subclip(starttime/40,starttime/40+duration)
clip2 = VideoFileClip("videos/pred_test_period.mp4")
clip3=VideoFileClip("videos/Manual_tail_test_period.mp4")
composite_clip = mp.CompositeVideoClip.CompositeVideoClip([clip1,clip2.set_position((0,0)),clip3.set_position((0,130))])
composite_clip.write_videofile("videos/classification_test_period.mp4", fps = 40)



