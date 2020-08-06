#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:29:42 2020

@author: ryan
"""



'''
#%%
#img_array is now point set of masks
#I try to filter the deeplabcut result based on whether they are inside the contour(dilated), but in the end I find 
#another way to calculate curvatue so it's not used at all
'''
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
#filtering only head, with extra measures
#Might need to implement filtering L/R eye at the same time as well
def head_inside_mask(df,mask_array,kernel_size=11,img_size=500):
    #columns:list of columns names we want to filter(['A_head','F_spine1'..]), if columns=all, then all columns are filtered
    data = df.copy()
    head_x=data.A_head.x.iloc[0]
    head_y=data.A_head.y.iloc[0]
    col="A_head"
    counter=0
    #this counter is a safety bell if at head_x just stuck at some very bad point or it's time recorded 
    #is too far away from the current time, and then it will just reset the "good" head position
    #Or maybe not resetting but just record this rare event so it can let L/R eye to work.
    for i in tqdm(range(df.shape[0])):
        x=min(img_size-1,int(np.round(df[col].x.iloc[i])))
        y=min(img_size-1,int(np.round(df[col].y.iloc[i])))
        mask=mask_array[i]
        #amplify the mask a little bit so the points near the contour are not ruled out
        k=cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
        eroded_mask=cv2.erode(mask, k, iterations=1)
        distance=np.sqrt(np.square((x-head_x))+np.square((y-head_y)))
        if eroded_mask[y,x]!=0:
            data[col].x.iloc[i]=np.nan
            data[col].y.iloc[i]=np.nan
            counter+=1
            #sometimes DLC misclassifies tail as head, so just to remove this by looking at the relative location changed
        elif distance>20 and counter<=7:
                #if cur point is close to the previous invalid pt, or cur point is far from prev valid pt
            data[col].x.iloc[i]=np.nan
            data[col].y.iloc[i]=np.nan
            counter+=1
        else:
            #record the latest valid_head position
            head_x=data[col].x.iloc[i]
            head_y=data[col].y.iloc[i]
            counter=0
    return data
#filtering wanted columns based on mask
def inside_mask(df,mask_array,columns="all",kernel_size=11,img_size=500):
    #columns:list of columns names we want to filter(['A_head','F_spine1'..]), if columns=all, then all columns are filtered
    data = df.copy()
    if columns=="all":
        columns=list(set(map(lambda x:x[0],df.columns)))
    for i in tqdm(range(df.shape[0])):
        for col in columns:
            x=min(img_size-1,int(np.round(df[col].x.iloc[i])))
            y=min(img_size-1,int(np.round(df[col].y.iloc[i])))
            mask=mask_array[i]
            #amplify the mask a little bit so the points near the contour are not ruled out
            k=cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
            eroded_mask=cv2.erode(mask, k, iterations=1)
            if eroded_mask[y,x]!=0:
                data[col].x.iloc[i]=np.nan
                data[col].y.iloc[i]=np.nan
    return data

#%%
#calculate the curvature at each point of the contour,default step=3

#currently only testing cosine angle
def compute_pointness(contour, n=3):
    out=np.zeros((600,600))
    contour=contour.squeeze()
    N = len(contour)
    t=1/N
    for i in range(N):
        x_cur, y_cur = contour[i]
        x_next, y_next = contour[(i + n) % N]
        x_prev, y_prev = contour[(i - n) % N]
        dy1=(y_next-y_prev)/(2*n*t)
        dx1=(x_next-x_prev)/(2*n*t)
        dy2=(y_next+y_prev-2*y_cur)/(n*n*t*t)
        dx2=(x_next+x_prev-2*x_cur)/(n*n*t*t)
        curvature=(dx1*dy2-dy1*dx2)/(np.power(dx1*dx1+dy1*dy1,3/2))
        curvature=abs(curvature)
        out[x_cur,y_cur]=curvature
    return out

#find the largest contour,will update it to main function later
def find_largest_contour(contours):
    flag=0
    area=0
    for cnt in contours:
        new_area=cv2.contourArea(cnt)
        if new_area>area and new_area<10000 :#in case some contour contains almost the whole image, not required if invert the image first
            area=new_area
            fish_contour=cnt
            flag=1
    if flag==1:
        return fish_contour,flag
    else:
        return np.zeros((1,1,2)),flag


def find_centroid(contour):
    moments=cv2.moments(contour)
    m01=moments.get("m01")
    m10=moments.get("m10")
    m00=moments.get("m00")
    xbar=m10/m00
    ybar=m01/m00
    return xbar,ybar

def compute_dist(contour,xbar,ybar):
    contour=contour.squeeze()
    dist=np.linalg.norm(contour-np.array([xbar,ybar]),axis=1)
    return dist


#give a heatmap along fish's contour about how curved it is
def plot_result(curvatures,contour,img_size=600,quantile=0.5):
    '''
    curvatures:the curvescore on the contour, should be an nxn array equalto img_size
    contour:one specific contour, nx1x2 array
    '''
    head_tail_x,head_tail_y=np.where(curvatures<0)
    new_curvatures=curvatures.copy()
    new_curvatures[new_curvatures<0]=0
    x,y=np.nonzero(new_curvatures)
    xbar,ybar=find_centroid(contour)
    colors=np.array(np.zeros_like(x),np.float64)
    dists=compute_dist(contour,xbar,ybar)
    for i in range(len(x)):
        colors[i]=new_curvatures[x[i],y[i]]
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.scatter(head_tail_x,head_tail_y,s=3,c="lemonchiffon")
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.2)#
    plt.plot(xbar,ybar,"ro",markersize=3,color="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.title("curveness heatmap on fish contour")
    plt.show()
    
def filter_curvature(curvatures,contour):
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    quantile=np.quantile(dists,0.6)
    contour=contour.squeeze()
    N=len(contour)
    
    for i in range(N):
        x,y=contour[i]
        if dists[i]>quantile:
            curvatures[x,y]=0.1
    return curvatures


#find the 4 boundary pts of the contour
#instead of using local flipping, use run length encoding to determine the longest 4 segments
def find_anchor_rle(contour,validity=None,quantile=0.5,min_length=100,warning_cnt=0):
    '''
    parameters:
        contour:fish's contour, likely an nx1x2 array
        validity: a nx2 boolean array which tells whether the contour point is valid(not too close to head/tail),
        if not provided, it will be autogenerated from contour
        quantile: the threshold to remove head/tail parts, every pts having distance to centroid larger than quantile(dist)
        will be deemed invalid
        warning_cnt:for debugging, since the function will only output 4indexs, it shall cnt for the times where it actually
        will find 3 more segments
    '''
    contour=contour.squeeze()
    #if the valid pts are given before hand, skip this step
    if validity is None or quantile!=0.5:
        xbar,ybar=find_centroid(contour)
        dists=compute_dist(contour,xbar,ybar)
        thres=np.quantile(dists,quantile)
        validity=dists<thres
    value,length=runLengthEncoding(validity)
   #squueze the rle lists by combining those shorter segments
    def shrink_rle(value,length):
        #this algo will just append those short segments to positive areas
        new_val=[]
        new_length=[]
        l=len(value)
        if value[0]!=value[l-1]:
            value.append(value[0])
            length.append(0)
        #if startpoint is actually at boundary, add a 0 length rle to it
        for i in range(l-1):
            if i==0:
                if (length[0]+length[l-1])>min_length:
                    new_val.append(value[0])
                else:
                    #say length starts like 1,2,1,1....2,1
                    new_val.append(-1)
                new_length.append(length[0])
            elif length[i]<min_length:
                #prev element in new_val is unknown or 1
                if new_val[::-1][0]!=0:
                    ll=len(new_length)-1
                    new_length[ll]+=length[i]
                else:
                    #if value negative, save this length to the next segment
                    new_val.append(-1)
                    new_length.append(length[i])
            else:
                ll=len(new_length)-1
                if new_val[::-1][0]==-1:
                    if value[i]==0: #saved length for previous, because it starts from 0
                        #so the previous segment is actually 1
                        new_val[ll]=True
                        new_val.append(False)
                        new_length.append(length[i])
                    else:
                        #the cur long segment is 1, so concat the previous short segments
                        new_val[ll]=1
                        new_length[ll]+=length[i]
                else:
                    if new_val[::-1][0]!=value[i]:
                        new_val.append(value[i])
                        new_length.append(length[i])
                    else:
                        new_length[ll]+=length[i]
            #put the last segment in
        ll=len(new_length)-1
        if new_val[ll]==-1:
            new_val[ll]=1
        if (length[0]+length[l-1])>min_length:
            new_val.append(value[l-1])
            new_length.append(length[l-1])
        else:
            #if the whole segment in the start point is too short, just give it to the last valid segment
            new_length[ll]+=length[l-1]
        return new_val,new_length
    new_val,new_length=shrink_rle(value,length)
    if len(new_val)<=3:
        warning_cnt+=1
        print("\r"+"less than 4 segments detected, already happens {} times".format(warning_cnt),end="")
        return np.nan,np.nan,np.nan,np.nan,warning_cnt
    elif new_val[0]:
        out2=new_length[0]
        in1=new_length[1]+new_length[0]
        out1=new_length[2]+new_length[1]+new_length[0]
        in2=min(len(validity)-1,new_length[3]+new_length[2]+new_length[1]+new_length[0])
    else:
        in1=new_length[0]
        out1=new_length[1]+new_length[0]
        in2=new_length[2]+new_length[1]+new_length[0]
        out2=min(len(validity)-1,new_length[3]+new_length[2]+new_length[1]+new_length[0])
    return in1,out1,in2,out2,warning_cnt

#compute the cos angle of a point on contour to its previous pt and next pt
def compute_cos(contour, step=3,img_size=600,min_step=2,quantile=0.5):
    #I plan to return 3 objects :an img_size x img_size array with each point having the cosine angle value on it,
    #  2 lists which is the cosine in left and right part
    out=np.zeros((img_size,img_size))
    left_cosines=[]
    right_cosines=[]
    contour=contour.squeeze()
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    quantile=np.quantile(dists,quantile)
    validity=dists<quantile
    in1,out1,in2,out2,_=find_anchor_rle(contour,validity)
    N = len(contour)
    t=1/N#actually no use cause the denomiator here cancelled out in later calculation
    def find_next(i,step,min_step):
        #find next point avaliable
        #say if index i+step is in the valid pts, then use i+step as tbe next point
        #otherwise backtrack the points until the point is valid or meet the minimal length requirement
        if step<=min_step or validity[(i+step)%N]==True:
            return contour[(i+step) % N]
        else:
            return find_next(i,step-1,min_step)
    def find_prev(i,step,min_step):
        if step<=min_step or validity[(i-step)%N]==True:
            return contour[(i -step) % N]
        else:
            return find_prev(i,step-1,min_step)
    for i in range(N):
        x_cur,y_cur=contour[i]
    for i in range(N):
        x_cur, y_cur = contour[i]
        if validity[i]==False:
            #head/tail position's cosine angle encoded to -1 for future visualization
            out[x_cur,y_cur]=-1
        else:
            x_next, y_next = find_next(i,step,min_step)
            x_prev, y_prev = find_prev(i,step,min_step)
            vec1=np.array([x_next-x_cur,y_next-y_cur])
            vec2=np.array([x_prev-x_cur,y_prev-y_cur])
            cos=np.sum(vec1*vec2)/np.sqrt(np.sum(vec1*vec1)*np.sum(vec2*vec2))
            out[x_cur,y_cur]=cos+1
            if i>in1 and i<out1:
                left_cosines.append(cos+1)
            else:
                right_cosines.append(cos+1)
    return out,np.array(left_cosines),np.array(right_cosines)

#just to visualize the distance for pt at n steps after and cur pt, parameter same as plot result
def visualize_steps(curvatures,contour,img_size=600,quantile=0.5,start=0,step=30):
    head_tail_x,head_tail_y=np.where(curvatures<0)
    new_curvatures=curvatures.copy()
    new_curvatures[new_curvatures<0]=0
    x,y=np.nonzero(new_curvatures)
    xbar,ybar=find_centroid(contour)
    colors=np.array(np.zeros_like(x),np.float64)
    dists=compute_dist(contour,xbar,ybar)
    for i in range(len(x)):
        colors[i]=new_curvatures[x[i],y[i]]
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.scatter(head_tail_x,head_tail_y,s=3,c="lemonchiffon")
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.2)#
    plt.plot(xbar,ybar,"ro",markersize=3,color="black")
    #this is the three points added
    x1=np.array([contour[start][:,0],contour[(start+step)%len(contour)][:,0],contour[(start-step)%len(contour)][:,0]]) 
    y1= np.array([contour[start][:,1],contour[(start+step)%len(contour)][:,1],contour[(start-step)%len(contour)][:,1]]) 
    plt.scatter(x1,y1,s=3,c="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.title("curveness heatmap on fish contour")
    plt.show()
   
    
#just to visualize the 4 boundary pt at fish contour, parameter same as plot result with in1,out1,in2,out2 be the boudary
#index of the contour, which should be in1-out1:first body part, in2-out2:second bodypart
def visualize_segments(curvatures,contour,in1,out1,in2,out2,img_size=600,quantile=0.5,to_array=False):
    head_tail_x,head_tail_y=np.where(curvatures<0)
    new_curvatures=curvatures.copy()
    new_curvatures[new_curvatures<0]=0
    x,y=np.nonzero(new_curvatures)
    xbar,ybar=find_centroid(contour)
    colors=np.array(np.zeros_like(x),np.float64)
    dists=compute_dist(contour,xbar,ybar)
    for i in range(len(x)):
        colors[i]=new_curvatures[x[i],y[i]]
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.scatter(head_tail_x,head_tail_y,s=3,c="lemonchiffon")
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.2)#
    plt.plot(xbar,ybar,"ro",markersize=3,color="black")
    #this is the three points added
    x1=np.array([contour[in1][:,0],contour[out1][:,0],contour[in2][:,0],contour[out2][:,0]]) 
    y1=np.array([contour[in1][:,1],contour[out1][:,1],contour[in2][:,1],contour[out2][:,1]]) 
    plt.scatter(x1,y1,s=3,c="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.title("curveness heatmap on fish contour")
    if to_array:
        out=mplfig_to_npimage(fig)
        plt.close(fig)
        return out
    else:
        plt.show()
  
#run length encoding:
#array: 1,0,0,0,1,1 -> length:1,3,2;value
def runLengthEncoding(arr): 
    value=[]
    length=[]
    prev=arr[0]
    value.append(prev)
    l=0
    for val in arr:
        if val==prev:
            l+=1
        else:
            length.append(l)
            value.append(val)
            l=1
        prev=val
    length.append(l)
    return value,length


'''
#find the flip point when going in/out from head/tail
#currently not the best way to find the boundary points
def find_anchor(contour,validity=None,min_body_length=150,min_head_length=50,quantile=0.5,warning_cnt=0,index=None):
    '''''''
    parameters:
        contour:fish's contour, likely an nx1x2 array
        validity: a nx2 boolean array which tells whether the contour point is valid(not too close to head/tail),
        if not provided, it will be autogenerated from contour
        min_body_length: a monitor parameter to ensure the length of the segment, so a flip in validity will not count if a recent
        flip already happend less than min_length pts before
        quantile: the threshold to remove head/tail parts, every pts having distance to centroid larger than quantile(dist)
        will be deemed invalid
        warning_cnt:for debugging, since the function will only output 4indexs, it shall cnt for the times where it actually
        will find 3 more segments
        
    '''''''
    #the 2 valid segment here should be in1-out1, in2-out2
    #return the index of the intersection point
    in1=-1
    out1=-1
    in2=-1
    out2=-1
    contour=contour.squeeze()
    #if the valid pts are given before hand, skip this step
    if validity is None or quantile!=0.5:
        xbar,ybar=find_centroid(contour)
        dists=compute_dist(contour,xbar,ybar)
        thres=np.quantile(dists,quantile)
        validity=dists<thres
    N = len(contour)
    start=validity[0]
    flag=validity[0]
    counter=max(min_head_length,min_body_length)+1
    if start==0:
        for i in range(N):
            #monitor the point where validity flips
            if flag==0 and validity[i]==1 and counter>min_head_length:
                if in1==-1:
                    in1=i
                elif in2==-1:
                    in2=i
                else:
                    warning_cnt+=1
                    if index is not None:
                        print("more than 2 segments detected, index is {} ".format(index))
                    else:
                        print("\r"+"more than 2 segments detected, already happens {} times".format(warning_cnt),end="")
                counter=0
                flag=1
            elif flag==1 and validity[i]==0 and counter>min_body_length:
                if out1==-1:
                    out1=i
                elif out2==-1:
                    out2=i
                flag=0
                counter=0
            else:
                counter+=1
                
    if start==1:
        for i in range(N):
            #monitor the point where validity flips
            if flag==0 and validity[i]==1 and counter>min_head_length:
                if in1==-1:
                    in1=i
                elif in2==-1:
                    in2=i
                else:
                    warning_cnt+=1
                    if index is not None:  
                        print("more than 2 segments detected, index is {} ".format(index))
                    else:
                        print("\r"+"more than 2 segments detected, already happens {} times".format(warning_cnt),end="")
                flag=1
                counter=0
            elif flag==1 and validity[i]==0 and counter>min_body_length:
                if out2==-1:
                    out2=i
                elif out1==-1:
                    out1=i
                flag=0   
                counter=0
            else:
                counter+=1
    if in1<0 or out1<0 or in2<0 or out2<0:
        print("not all four anchors are detected for index"+str(index))
    return in1,out1,in2,out2,warning_cnt
'''
