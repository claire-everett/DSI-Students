#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:44:46 2020

@author: ryan
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from contour_utils import find_centroid
IMGSIZE=1500
with open("data/IM1_IM2_2.1.1_L_70000_150000_head_index", "rb") as fp:  
    head_indexs = pickle.load(fp)

with open("data/IM1_IM2_2.1.1_L_70000_150000_tail_index", "rb") as fp:   
    tail_indexs = pickle.load(fp)
with open("data/IM1_IM2_2.1.1_L_70000_150000_fish_segment_length", "rb") as fp:  
    lengths = pickle.load(fp)
with open("data/IM1_IM2_2.1.1_L_70000_150000_curve_scores", "rb") as fp:   
    curve_scores = pickle.load(fp)  
with open("data/IM1_IM2_2.1.1_L_70000_150000_intersections", "rb") as fp:   
    intersections = pickle.load(fp)  
   
with open("data/contour_array", "rb") as fp:   
    new_contour_array = pickle.load(fp)
    
curve_score=curve_scores[0]
head_index=head_indexs[0]
tail_index=tail_indexs[0]
intersect=intersections[0]
#visualize the contour with the curveness as heatmap, and head and tail on it
fig=plt.figure()
contour=curve_score[:,:2]      
plt.scatter(curve_score[:,0], curve_score[:,1], c=curve_score[:,2], s=3,cmap="YlGnBu", vmin=0, vmax=1)
plt.plot(np.float64(contour[tail_index,0]),np.float64(contour[tail_index,1]),"ro",markersize=5,color="red")
plt.plot(np.float64(contour[head_index,0]),np.float64(contour[head_index,1]),"ro",markersize=5,color="purple")
plt.xlim(0,1500)
plt.ylim(0,1500)


#visualize the distance from tail to the main body line
fig=plt.figure()      
plt.scatter(curve_score[:,0], curve_score[:,1], c=curve_score[:,2], s=3,cmap="YlGnBu", vmin=0, vmax=1)
plt.plot(np.float64(contour[tail_index,0]),np.float64(contour[tail_index,1]),"ro",markersize=5,color="red")
plt.plot(np.float64(contour[head_index,0]),np.float64(contour[head_index,1]),"ro",markersize=5,color="purple")
head=contour[head_index]
tail=contour[tail_index]  
plt.plot([intersect[0],head[0]],[intersect[1],head[1]],linewidth=1, linestyle = "-", color="red")
plt.plot([intersect[0],tail[0]],[intersect[1],tail[1]],linewidth=1, linestyle = "-", color="red")
plt.plot(intersect[0],intersect[1],"ro",markersize=5,color="red")
plt.xlim(0,IMGSIZE)
plt.ylim(0,IMGSIZE)
plt.colorbar()

contour=curve_score[:,:2]
N=len(contour)
fig=plt.figure()      
plt.scatter(curve_score[:,0], curve_score[:,1], c=curve_score[:,2], s=3,cmap="YlGnBu", vmin=0, vmax=1)
plt.plot(np.float64(contour[tail_index,0]),np.float64(contour[tail_index,1]),"ro",markersize=5,color="red")
plt.plot(np.float64(contour[head_index,0]),np.float64(contour[head_index,1]),"ro",markersize=5,color="purple")
head=contour[head_index]
tail=contour[tail_index]
centroid=find_centroid(contour.reshape((N,1,2)).astype(int))
tail_mid=(contour[(tail_index+100)%N]+contour[(tail_index-100)%N])/2
head_mid=(contour[(head_index+150)%N]+contour[(head_index-150)%N])/2
#100 pixel length with direction same as tail/head
pt1=centroid+50/np.linalg.norm(np.array([head[0]-head_mid[0],head[1]-head_mid[1]]))*np.array([head[0]-head_mid[0],head[1]-head_mid[1]])
pt2=centroid+50/np.linalg.norm(np.array([tail[0]-tail_mid[0],tail[1]-tail_mid[1]]))*np.array([tail[0]-tail_mid[0],tail[1]-tail_mid[1]])   
plt.plot([centroid[0],pt1[0]],[centroid[1],pt1[1]],linewidth=1, linestyle = "-", color="red")
plt.plot([centroid[0],pt2[0]],[centroid[1],pt2[1]],linewidth=1, linestyle = "-", color="red")
plt.plot(centroid[0],centroid[1],"ro",markersize=5,color="red")
plt.xlim(0,IMGSIZE)
plt.ylim(0,IMGSIZE)
plt.colorbar()




