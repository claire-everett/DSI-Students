#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:39:13 2020

@author: ryan
"""
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.ndimage import zoom  
from contour_utils import *
from find_features import features
from scipy import sparse
from LoopingArray import LoopingArray
import pickle
#vidcap = cv2.VideoCapture("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4")
#success,image=vidcap.read()

IMAGELENGTH=500
fps=40
videopath="TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4"

#find a conservative mask to exclude reflections and noises
def find_conservative_mask(videopath,length=40*300,start=0,step=1,pre_filter=None):
    '''
    prefilter:a function that changes mask array
    length: Total number of frames needed
    start: which frame to start
    step:pick every * frame
    '''
    vidcap = cv2.VideoCapture(videopath)
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
        #filter specific large black area, like 888 sign in other videos
        if pre_filter is not None:
            th=pre_filter(th)
        contours, hierarchy = cv2.findContours(np.invert(th.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#simplified contour pts
        #finding the contour with largest area
        fish_contour,flag=find_largest_contour(contours)
        if flag==0:
            print("no valid fish contour find at index {}".format(i))
            img=np.float32(np.full(image.shape,0))#just in case there's no valid contour, won't happen in the current case
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
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1]),255)),[true_contour],0,(0,0,0),cv2.FILLED),np.uint8))
    return mask_array,contour_array

#add tail to previous mask
def find_tail(videopath,conservative_masks,conservative_contours,head,interpolate=True,start=0,step=1,pre_filter=None):
    '''
    Parameters
    ----------
    conservative_masks : mask derived from the previous step
    conservative_contours : contour derived from the previous step
    head : head location from DLC
    interpolate :  optional,whether to interpolate the image to smooth it

    '''
    vidcap = cv2.VideoCapture(videopath)
    contour_array=[]
    index=start
    vidcap.set(1,index)
    length=len(conservative_masks)
    def exclusion(y,x):
        dists1=compute_dist(np.vstack([x,y]).T,xbar,ybar)
        #new found part should not be too close to the centroid(as it is tail)
        flag1=dists1<thres
        dists2=compute_dist(np.vstack([x,y]).T,head_x,head_y)
        #new found part shall not be too close to the head, otherwise it is likely a reflection
        flag2=dists2<thres
        flag=np.logical_or(flag1,flag2)
        x=x[~flag]
        y=y[~flag]
        data=np.full((len(x),),255)
        output=np.invert(np.uint8(sparse.coo_matrix((data,(y,x)),shape=(IMAGELENGTH,IMAGELENGTH)).toarray()))
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
        if flag==0:
            print("no valid fish contour find at index {}".format(index-1))
            img=np.float32(np.full(image.shape,0))#just in case there's no valid contour, won't happen in the current case
        else:
            img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
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
    return contour_array
    

def head_on_contour(head_x,head_y,contour):
    dists=compute_dist(contour,head_x,head_y)
    index=np.argmin(dists)
    return index

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

def averageBlur(arr,neighbor_width=30):
    #updating an array of curviness, make every element to the avg valye of its neighbor
    #So the tail tip is more likely to get a higher score compared to its neighbors
    out=arr.copy()
    l=len(arr)
    for i in range(l):
        out[i]=np.sum(LoopingArray(arr)[(i-neighbor_width):i+neighbor_width+1])/(2*neighbor_width+1)
    return out

#after the runlengthEncoding on validity derived from distance to centroid,
#some segments should be combined into longer segments
def combine_small_segment(value,length,minimal_length=70):
    l=len(value)
    flag=0
    #check if the starting point is a flipping point
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
        #if the segment is smaller than certain length, put it to a existing segment
        #if there is no such thing, prepare that for a future valid segment
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
        #put the length back when we first cut off the last piece at first
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
            return np.nan,np.nan,np.nan,np.nan
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
    when slicing/getting items when keep using modulus
    '''
    if np.isnan(tail_start):
        return np.nan,np.nan,curviness_score_tail,length
    tail_segment=LoopingArray(blurred_curviness)[tail_start:tail_end+1]
    tail_index=(np.argmax(tail_segment)+tail_start)%l
    curviness=curviness_score_head[:,2]
    blurred_curviness=averageBlur(curviness,neighbor_width[1])
    head_segment=LoopingArray(blurred_curviness)[head_start:head_end+1]
    better_head_index=(np.argmax(head_segment)+head_start)%l
    return better_head_index,tail_index,curviness_score_tail,length

#find the raw mask with tail likely excluded
mask_array,contour_array=find_conservative_mask("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4",length=80000,start=70000,step=1)
print("complete extract raw mask")
#save mask and contour information
with open("data/IM1_IM2_2.1.1_L_70000_150000_mask_raw", "wb") as fp:
    pickle.dump(mask_array, fp)
    
with open("data/IM1_IM2_2.1.1_L_70000_150000_contour_raw", "wb") as fp:
    pickle.dump(contour_array, fp)

#get a better contour using head position from DLC and thresholding again.
#the previous mask is also needed to remove some noise near fish's contour caused by raising the threshold
path = "h5files/h5 2/IM1_IM2_2.1.1_LDLC_resnet50_DLC_toptrackFeb27shuffle1_170000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
df=df.iloc[70000:150000,:]
new_features=features(starttime=0,endtime=80000)
filtered_df=new_features.filter_df(df,add_midpoint=True)
filtered_head=relative_position_check(filtered_df.A_head)
filtered_head=filtered_head.fillna(method="ffill")
new_contour_array=find_tail("TailBeatingExamples/Copy of IM1_IM2_2.1.1_L.mp4",mask_array,contour_array,filtered_head,start=70000,step=1,interpolate=True)

#save more accurate contour
with open("data/IM1_IM2_2.1.1_L_70000_150000_contour_refined", "wb") as fp:
    pickle.dump(new_contour_array, fp)
print("complete extract accurate mask")   


#find head and tail position
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
    
with open("data/IM1_IM2_2.1.1_L_70000_150000_curve_scores", "wb") as fp:
    pickle.dump(curve_scores, fp)
    
with open("data/IM1_IM2_2.1.1_L_70000_150000_tail_index", "wb") as fp:
    pickle.dump(tail_indexs, fp)   
    
with open("data/IM1_IM2_2.1.1_L_70000_150000_head_index", "wb") as fp:
    pickle.dump(better_head_indexs, fp)    

with open("data/IM1_IM2_2.1.1_L_70000_150000_fish_segment_length", "wb") as fp:
    pickle.dump(lengths, fp)       
    
print("tail feature extracted!")

head_indexs=pd.Series(better_head_indexs).fillna(method="ffill").astype(int)
tail_indexs=pd.Series(tail_indexs).fillna(method="ffill").astype(int)

head_midlines=[]
tail_midlines=[]
for i in tqdm(range(len(head_indexs))):
    contour=curve_scores[i][:,:2]
    contour=contour.squeeze()
    N=len(contour)
    head_index=head_indexs[i]
    tail_index=tail_indexs[i]
    midline=np.zeros((0,2),dtype=np.float64)
    #only use the starting point and the 150th point after in both direction
    #can use lr if not good enough
    for j in [0,150]:
        midline=np.concatenate((midline,((contour[(head_index+j)%N]+contour[(head_index-j)%N])/2).reshape(1,2)),0)
    head_midlines.append(midline) 
    midline=np.zeros((0,2),dtype=np.float64)
    for j in [0,100]:
        midline=np.concatenate((midline,((contour[(tail_index+j)%N]+contour[(tail_index-j)%N])/2).reshape(1,2)),0)
    tail_midlines.append(midline) 
    
tail_angle=[]
#get a line starting from tail, a line starting from head, calculate cross angle
for i in tqdm(range(len(head_indexs))):
    head_midline=head_midlines[i]
    tail_midline=tail_midlines[i]
    vec1=np.array(head_midline[0]-head_midline[1])
    vec2=np.array(tail_midline[0]-tail_midline[1])
    cos=np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    angel=np.arccos(cos)/np.pi*180
    tail_angle.append(angel)

distance_to_spine=[]
intersections=[]
#calculate the distance from tail to the line starting from head
#intersection is only used for visualization
for i in range(len(head_indexs)):
    head_midline=head_midlines[i]
    tail_midline=tail_midlines[i]
    headx1,heady1=head_midline[0]
    headx2,heady2=head_midline[1]
    #headx2,heady2=head_midline[1]
    tailx,taily=tail_midline[0]
    if headx1==headx2:
        d=abs(tailx-headx1)
        k1=10000
        b1=-k1*headx1
    elif heady1==heady2:
        d=abs(taily-heady1)
        k1=0
        b1=heady1
    else:
        k1=(heady1-heady2)/(headx1-headx2)
        b1=heady2-k1*headx2
        d=abs(k1*tailx-taily+b1)/np.sqrt(k1**2+1)
    distance_to_spine.append(d)
    try:
        k2=-1/k1
        b2=taily-tailx*k2
    except ZeroDivisionError:
        k2=10000
        b2=-k2*tailx      
    try:
        x=(b2-b1)/(k1-k2)
        intersect=(x,k1*x+b1)
        intersections.append(intersect)
    except ZeroDivisionError:
        #use midpoint of head and tail, assume it's just the same line
        intersections.append(((headx1+tailx)/2,(heady1+taily)/2))

with open("data/IM1_IM2_2.1.1_L_70000_150000_tailAngle", "wb") as fp:
    pickle.dump(tail_angle, fp)    

with open("data/IM1_IM2_2.1.1_L_70000_150000_distance_to_spineline", "wb") as fp:
    pickle.dump(distance_to_spine, fp)  
with open("data/IM1_IM2_2.1.1_L_70000_150000_intersections", "wb") as fp:
    pickle.dump(intersections, fp)  