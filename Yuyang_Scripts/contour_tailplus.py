#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:37:48 2020

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
from contour_utils import *
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
from scipy import sparse
from LoopingArray import LoopingArray
#vidcap = cv2.VideoCapture("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4")
#success,image=vidcap.read()

image_length=500
fps=40
videopath="TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"
def find_conservative_mask(videopath,length=40*300,start=0,step=1,pre_filter=None):
    '''
    prefilter:a function that changes mask array
    '''
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
        #th=cv2.GaussianBlur(th, (3, 3), 0)
        #first shrink the mask to get it separate from its reflection
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        th = cv2.dilate(th, k2, iterations=1)
        #filter specific large black area
        if pre_filter is not None:
            th=pre_filter(th)
        contours, hierarchy = cv2.findContours(np.invert(th.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#simplified contour pts
        #finding the contour with largest area
        fish_contour,flag=find_largest_contour(contours)
        if flag==0:
            print("no valid fish contour find at index {}".format(i))
            img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
        else:
            #draw only the mask of this contour
            img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img=np.invert(np.array(img,np.uint8))
        #zoom the mask a little bit because we shrink it before
        k3=cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
        img=np.invert(np.array(cv2.erode(img,k3,iterations=1),np.uint8))
        #smoothing
        img=cv2.medianBlur(img, 5)
        true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#complete contour pts
        true_contour,_=find_largest_contour(true_contour)
        try:
            contour_array.append(true_contour)
        except:
            contour_array.append(np.zeros((1,2),dtype=np.float64))
        img_array.append(np.array(cv2.drawContours(image.copy(),[true_contour],0,(0,255,0),2),np.uint8))
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1]),255)),[true_contour],0,(0,0,0),cv2.FILLED),np.uint8))
    return img_array,mask_array,contour_array

img_array,mask_array,contour_array=find_conservative_mask("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4",length=5000,start=90000,step=1)
out = cv2.VideoWriter('videos/IM1_IM2_conservativemask_erode5_test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(img_array)):
     out.write(img_array[i])
out.release()



samples=random.randint(0,2000,5)
for sample in samples:
    plt.figure()
    plt.imshow(img_array[sample])

'''
plt.imshow(img_array[0])
plt.axhline(450)
plt.axhline(85)
plt.axvline(100)
plt.axvline(490)
'''

def find_tail(videopath,conservative_masks,conservative_contours,head,interpolate=True,start=0,step=1,pre_filter=None):
    vidcap = cv2.VideoCapture(videopath)
    img_array=[]
    mask_array=[]
    contour_array=[]
    index=start
    vidcap.set(1,index)
    length=len(conservative_masks)
    def exclusion(y,x):
        #flag1=np.logical_or(y>450,y<85)
        #flag2=np.logical_or(x<100,x>485)
        dists1=compute_dist(np.vstack([x,y]).T,xbar,ybar)
        #new found part should not be too close to the centroid(as it is tail)
        flag1=dists1<thres
        dists2=compute_dist(np.vstack([x,y]).T,head_x,head_y)
        #new found part shall not be too close to the head, otherwise it is likely be reflection
        '''
        what if the head DLC find is actually not accurate even after filtering?
        idea1: Considering left eye and right eye as well, it the filtering function filters out too many pts in a row,
        just let it deem next n head pts as nan instead and then resume the normal process, and in the mean time the
        we only check distance between mask and L/R eye.
        idea2:use the previous filtering method instead?
        '''
        flag2=dists2<thres
        flag=np.logical_or(flag1,flag2)
        x=x[~flag]
        y=y[~flag]
        data=np.full((len(x),),255)
        output=np.invert(np.uint8(sparse.coo_matrix((data,(y,x)),shape=(image_length,image_length)).toarray()))
        return output    
    for i in tqdm(range(length)):
        success,image=vidcap.read()
        if success!=1:
            print("process stops early at {}th iteration".format(i))
            break
        index+=step
        vidcap.set(1,index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        th = cv2.erode(th, k1, iterations=1)
        #filtering the mask derived by direct thresholding using the previous derived conservative mask and head position
        conservative_contour=conservative_contours[i]
        conservative_mask=conservative_masks[i]
        xbar,ybar=find_centroid(conservative_contour)
        dists=compute_dist(conservative_contour,xbar,ybar)
        thres=np.quantile(dists,0.5)
        y,x=np.nonzero(np.invert(th))#black area, i.e. fish body
        head_x=head.x.iloc[i]
        head_y=head.y.iloc[i]
        th=exclusion(y,x)           
        th=np.minimum(th,conservative_mask)
        if pre_filter is not None:
            th=pre_filter(th)
        th=np.invert(th)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fish_contour,flag=find_largest_contour(contours)
        '''
        flag=0
        area=0
        for cnt in contours:
            new_area=cv2.contourArea(cnt)
            if new_area>area and new_area<10000 :#in case some contour contains almost the whole image, not required if invert the image first
                area=new_area
                fish_contour=cnt
                flag=1
            '''
        if flag==0:
            print("no valid fish contour find at index {}".format(i))
            img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
            img_array.append(img)
        else:
            img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
            img_array.append(np.array(cv2.drawContours(image.copy(),[fish_contour],0,(0,255,0),2),np.uint8))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #to get rid of small holes in the mask
        k3=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        img=np.invert(np.array(cv2.erode(img,k3,iterations=1),np.uint8))
        if interpolate==True:
            img=zoom(img,3)
            img=cv2.medianBlur(img, 21) #since zoom made the img larger
        else:
             img=cv2.medianBlur(img, 5)
        true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#complete contour pts
        true_contour,_=find_largest_contour(true_contour,interpolate=1)
        try:
            contour_array.append(true_contour)
        except:
            contour_array.append(np.zeros((1,2),dtype=np.float64))
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1]),255)),[true_contour],0,(0,0,0),cv2.FILLED),np.uint8))
    return img_array,mask_array,contour_array
   

path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
df=df.iloc[90000:95000,:]
new_features=features(starttime=0,endtime=5000)
filtered_df=new_features.filter_df(df,add_midpoint=True)
#filtered_df=inside_mask(df,mask_array,kernel_size=11)
#filtered_df=filtered_df.fillna(method="ffill")
filtered_head=relative_position_check(filtered_df.A_head)
filtered_head=filtered_head.fillna(method="ffill")
    
#head=filtered_df.A_head
head_x=filtered_head.x
head_y=filtered_head.y
'''
head_x=filtered_df.A_head.x
head_y=filtered_df.A_head.y 
spine1_x=filtered_df.F_spine1.x
spine1_y=filtered_df.F_spine1.y
Roper_x=filtered_df.B_rightoperculum.x
Roper_y=filtered_df.B_rightoperculum.y
Loper_x=filtered_df.E_leftoperculum.x
Loper_y=filtered_df.E_leftoperculum.y
'''
'''
head_x=df.A_head.x
head_y=df.A_head.y 
spine1_x=df.F_spine1.x
spine1_y=df.F_spine1.y
Roper_x=df.B_rightoperculum.x
Roper_y=df.B_rightoperculum.y
Loper_x=df.E_leftoperculum.x
Loper_y=df.E_leftoperculum.y
'''

new_img_array,new_mask_array,new_contour_array=find_tail("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4",mask_array,contour_array,filtered_head,start=90000,step=1,interpolate=True)
out = cv2.VideoWriter('videos/test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(new_img_array)):
     out.write(new_img_array[i])
out.release()


'''
for i in range(1560,1600):
    plt.figure()
    plt.imshow(new_img_array[i])
'''
#plt.plot(x[542],y[354],"ro",markersize=2)


#this video is to look at whether the filer on head is good
def make_frame(t):
    time=int(t*fps)
    mask=mask_array[time]
    x=head_x.iloc[time]
    y=head_y.iloc[time]
    
    sx=spine1_x[time]
    sy=spine1_y[time]
    lox=Loper_x[time]
    loy=Loper_y[time]
    rox=Roper_x[time]
    roy=Roper_y[time]
    
    fig=plt.figure()
    plt.imshow(mask,"gray")
    plt.plot(x,y,"ro",markersize=3)
    
    plt.plot(sx,sy,"ro",markersize=3,c="blue")
    plt.plot(lox,loy,"ro",markersize=3,c="blue")
    plt.plot(rox,roy,"ro",markersize=3,c="blue")
    plt.text(-150,50,"{} frame".format(time),bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    plt.xlim(0,500)
    plt.ylim(0,500)
    out=mplfig_to_npimage(fig)
    plt.close(fig)
    return out

animation = VideoClip(make_frame, duration = 125)
animation.write_videofile("videos/head_operculum_spine1_raw.mp4", fps=fps)

def head_on_contour(head_x,head_y,contour):
    dists=compute_dist(contour,head_x,head_y)
    index=np.argmin(dists)
    return index
    
midlines=[]
for i in tqdm(range(len(new_contour_array))):
    contour=new_contour_array[i]
    contour=contour.squeeze()
    N=len(contour)
    x=head_x[i]
    y=head_y[i]
    head_index=head_on_contour(x*3,y*3,contour)
    midline=np.zeros((0,2),dtype=np.float64)
    for j in range(int(N/2)):
        midline=np.concatenate((midline,((contour[(head_index+j)%N]+contour[(head_index-j)%N])/2).reshape(1,2)),0)
    midlines.append(midline) 

plt.imshow(new_mask_array[0])
plt.plot(head_x[0]*3,head_y[0]*3,"ro")

def visualize_midlines(t):
    time=int(t*40)
    contour=new_contour_array[time]
    midline=midlines[time]
    contour=contour.squeeze()
    fig=plt.figure(dpi=100)
    x=contour[:,0]
    y=contour[:,1]
    plt.scatter(x,y,s=0.1,color="lemonchiffon")
    xmid=midline[:,0]
    ymid=midline[:,1]
    plt.scatter(xmid,ymid,s=0.1,color="blue")
    plt.xlim(0,1800)
    plt.ylim(0,1800)
    out=mplfig_to_npimage(fig)
    plt.close(fig)
    return out
    
animation = VideoClip(visualize_midlines, duration = 30)
animation.write_videofile("videos/midlines_test.mp4", fps=40)

def compute_cos_fullbody(contour, step=30):
    #compute the curvature for full body
    curviness=np.zeros((0,3),dtype=np.float64)
    contour=contour.squeeze()
    N = len(contour)
    def find_next(i,step):
        return contour[(i+step) % N]
    def find_prev(i,step):
        return contour[(i -step) % N]
    for i in range(N):
        x_cur, y_cur = contour[i]
        x_next, y_next = find_next(i,step)
        x_prev, y_prev = find_prev(i,step)
        vec1=np.array([x_next-x_cur,y_next-y_cur])
        vec2=np.array([x_prev-x_cur,y_prev-y_cur])
        cos=np.sum(vec1*vec2)/np.sqrt(np.sum(vec1*vec1)*np.sum(vec2*vec2))
        curviness=np.concatenate((curviness,np.array([x_cur,y_cur,cos+1]).reshape(1,3)))
    return curviness

curve_scores=[]
head_index=[]
for i in tqdm(range(len(new_contour_array))):
    contour=new_contour_array[i]
    curve_scores.append(compute_cos(contour,step=100))
    head_index.append(head_on_contour(head_x[i],head_y[i],contour))

def plot_result(curvatures,contour,length,tail_index,head_index,time,midline=None,img_size=600,quantile=0.5,to_array=False,vmax=1):
    '''
    curvatures:the curvescore on the contour, should be an nxn array equalto img_size
    contour:one specific contour, nx1x2 array
    for plotting tail and the flipping pts, for debugging
    '''
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.scatter(curvatures[:,0], curvatures[:,1], c=curvatures[:,2], s=3,cmap="YlGnBu", vmin=0, vmax=vmax)#
    x1=contour[(np.cumsum(length)-1)[:-1]][:,:,0]
    y1=contour[(np.cumsum(length)-1)[:-1]][:,:,1]
    plt.plot(np.float64(contour[tail_index,0,0]),np.float64(contour[tail_index,0,1]),"ro",markersize=5,color="red")
    plt.plot(np.float64(contour[head_index,0,0]),np.float64(contour[head_index,0,1]),"ro",markersize=5,color="purple")
    plt.scatter(x1,y1,s=3,c="green")
    if midline is not None:
         xmid=midline[:,0]
         ymid=midline[:,1]
         plt.scatter(xmid,ymid,s=0.1,color="blue")
    plt.plot(xbar,ybar,"ro",markersize=3,color="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.text(-300,80,"{} frame".format(time),bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    plt.title("curveness heatmap on fish contour")
    if not to_array:
        plt.show()
    else:
        out=mplfig_to_npimage(fig)
        plt.close(fig)
        return out
    
plt.plot(curve_scores[0][:,2])
plt.axvline(x=head_index[0],c="red")
plt.axvline(x=head_index[0]+int(len(curve_scores[0])/2),c="red")


def averageBlur(arr,neighbor_width=30):
    #updating an array of curviness, make every element to the avg valye of its neighbor
    #So the tail tip is more likely to get a higher score compared to its neighbors
    out=arr.copy()
    l=len(arr)
    for i in range(l):
        out[i]=np.sum(LoopingArray(arr)[(i-neighbor_width):i+neighbor_width+1])/(2*neighbor_width+1)
    return out

averageBlur(curve_scores[0][:,2])
test=curve_scores[0].copy()
test[:,2]=averageBlur(test[:,2],neighbor_width=20)
plot_result(test,new_contour_array[0],img_size=1500,vmax=1)


def combine_small_segment(value,length,minimal_length=70):
    l=len(value)
    flag=0
    if value[0]==value[l-1]:
        temp=length[l-1]
        flag=1
        length[0]+=temp
        value=value[:l-1]
        length=length[:l-1]
    new_val=[]
    new_length=[]
    stack=0
    for i in range(len(value)):
        ll=length[i]
        if ll<minimal_length:
            if new_val:                
                new_length[len(new_length)-1]+=ll
            else:
                stack+=ll
        else:
            if len(new_val)==0 or new_val[::-1][0]!=value[i]:
                new_val.append(value[i])
                new_length.append(length[i]+stack)
                stack=0
            else:
                new_length[len(new_length)-1]+=ll
    if flag==1:
        if new_val[len(new_length)-1]!=new_val[0]:
            new_val.append(new_val[0])
            new_length.append(temp)
        else:
            new_length[len(new_length)-1]+=temp
        new_length[0]-=temp
            
    return np.array(new_val),np.array(new_length)

def predict_tail(contour,head_index,step=None,quantile=0.4,neighbor_width=None):
    #default value, in case the image is/is not zoomed
    if step is None:
        step=[int(len(contour)/10),int(len(contour)/10)*1.5]
    if neighbor_width is None:
        neighbor_width=[int(len(contour)/20),int(len(contour)/20)]
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    thres=np.quantile(dists,quantile)
    #get whether the pts are far from the centroid
    validity=dists>thres
    l=len(dists)
    value,length=runLengthEncoding(validity)
    value,length=combine_small_segment(value, length)
    def find_segment(value,length):
        #index of segment
        l=len(value)
        new_length=length.copy()
        new_value=value.copy()
        #append the last segment unfinished segment to the front to compare the length
        if value[l-1]==value[0]:
            new_length[0]=length[0]+length[l-1]
            new_value=value[:-1]
            new_length=new_length[:-1]
        try:
            #if the filtering goes wrong and less than 2 positive segments were found
            longest,second_longest=np.sort(new_length[new_value==1])[::-1][:2]
        except:
            print("less than 2 segments found at index{}".format(i))
            return np.nan
        #finding the longest 2 segments which are far from centroid
        if np.where(np.logical_and(new_value==1,new_length==longest))[0].shape[0]==1:
            longest_positive_index=np.where(np.logical_and(new_value==1,new_length==longest))[0][0]
            second_longest_positive_index=np.where(np.logical_and(new_value==1,new_length==second_longest))[0][0]
        else:
            #in case the 2 segments are of equal length
            longest_positive_index=np.where(np.logical_and(new_value==1,new_length==longest))[0][0]
            second_longest_positive_index=np.where(np.logical_and(new_value==1,new_length==second_longest))[0][1]
        if longest_positive_index!=0:
            longest_positive_interval=[np.sum(length[:longest_positive_index]),np.sum(length[:longest_positive_index+1])-1]
        else:
            longest_positive_interval=[np.sum(length[:-1])%len(contour),length[0]-1]
        if second_longest_positive_index!=0:
            second_longest_positive_interval=[np.sum(length[:second_longest_positive_index]),np.sum(length[:second_longest_positive_index+1])-1]
        else:
            second_longest_positive_interval=[np.sum(length[:-1])%len(contour),length[0]-1]
        if (head_index<=longest_positive_interval[1] and 
            head_index>=longest_positive_interval[0]) or (head_index>=longest_positive_interval[0] 
            and longest_positive_interval[0]>longest_positive_interval[1]):
            #if head in the first segment,return the second segment
            return second_longest_positive_interval[0],second_longest_positive_interval[1],longest_positive_interval[0],longest_positive_interval[1]
        elif (head_index<=second_longest_positive_interval[1] and 
              #if head in the second segment,return the first segment
            head_index>=second_longest_positive_interval[0]) or (head_index>=second_longest_positive_interval[0] 
            and second_longest_positive_interval[0]>second_longest_positive_interval[1]):
            return longest_positive_interval[0],longest_positive_interval[1],second_longest_positive_interval[0],second_longest_positive_interval[1]
        else:
            #if head index not inside the valid segment when the fish is too curved?
            #choose tail segment as the segment furthur from the head_index
            dist_to_longest=min((head_index-longest_positive_interval[0])%len(contour),(head_index-longest_positive_interval[1])%len(contour),
                                (longest_positive_interval[0]-head_index)%len(contour),(longest_positive_interval[1]-head_index)%len(contour))
            dist_to_second_longest=min((head_index-second_longest_positive_interval[0])%len(contour),(head_index-second_longest_positive_interval[1])%len(contour),
                                (second_longest_positive_interval[0]-head_index)%len(contour),(second_longest_positive_interval[1]-head_index)%len(contour))
            if dist_to_second_longest>=dist_to_longest:
                return second_longest_positive_interval[0],second_longest_positive_interval[1],longest_positive_interval[0],longest_positive_interval[1]
            else:
                return longest_positive_interval[0],longest_positive_interval[1],second_longest_positive_interval[0],second_longest_positive_interval[1]
            
    tail_start,tail_end,head_start,head_end=find_segment(value,length)
    curviness_score_tail=compute_cos_fullbody(contour,step=step[0])
    curviness_score_head=compute_cos_fullbody(contour,step=step[1])
    curviness=curviness_score_tail[:,2]
    blurred_curviness=averageBlur(curviness,neighbor_width[0])
    '''
    sorry for using this imcomplete self defined class, i am just too confused by the ring structure of the contour
    when slicing/getting items when keep using modulus, will revise later
    '''
    tail_segment=LoopingArray(blurred_curviness)[tail_start:tail_end+1]
    tail_index=(np.argmax(tail_segment)+tail_start)%l
    curviness=curviness_score_head[:,2]
    blurred_curviness=averageBlur(curviness,neighbor_width[1])
    head_segment=LoopingArray(blurred_curviness)[head_start:head_end+1]
    better_head_index=(np.argmax(head_segment)+head_start)%l
    return better_head_index,tail_index,curviness_score_tail,length
            
curve_scores=[]
tail_indexs=[]
better_head_indexs=[]
lengths=[]
for i in tqdm(range(len(new_contour_array))):
    contour=new_contour_array[i]
    #since image is interpolated, and its size is 3 times before
    head_index=head_on_contour(head_x[i]*3, head_y[i]*3, contour)
    better_head_index,tail_index,curve_score,length=predict_tail(contour,head_index,step=[100,170],neighbor_width=[50,50])
    tail_indexs.append(tail_index)
    curve_scores.append(curve_score)
    lengths.append(length)
    better_head_indexs.append(better_head_index)
                
head_indexs=[]
for i in tqdm(range(len(new_contour_array))):
    contour=new_contour_array[i]
    #since image is interpolated, and its size is 3 times before
    head_indexs.append(head_on_contour(head_x[i]*3, head_y[i]*3, contour))


def make_frame(t):
    time=int(t*40)
    contour=new_contour_array[time]
    curviness=curve_scores[time]
    tail_index=tail_indexs[time]
    length=lengths[time]
    head_index=better_head_indexs[time]
    return plot_result(curviness,contour,length,tail_index,head_index,time,quantile=0.4,img_size=1500,to_array=True,vmax=1)                

animation = VideoClip(make_frame, duration = 125)
animation.write_videofile("videos/tail_extract_test8.mp4", fps=40)

            



