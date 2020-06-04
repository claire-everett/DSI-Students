#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:10:17 2020

@author: miaoyuqi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import *
from scipy import interpolate
# firstly, filter data for tail beating
def filter_tailbeating(data,p0=50,p1=15,t1 = 20):
    # Yuyang's method
    # check points location intervals
    mydata = data.copy()
#     boi = ['A_head','B_rightoperculum','E_leftoperculum',"F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7",'C_tailbase']
#     for b in boi:
#         for j in ['x','y']:
#             xdifference = abs(mydata[b][j].diff())
#             xdiff_check = xdifference > p0     
#             mydata[b][j][xdiff_check] = np.nan
    spine_column=["A_head","F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7"] #,'D_tailtip','C_tailbase']
    for i,c in enumerate(spine_column):
        # using the spine1 as the original points
        if i == 0:
            dist = np.sqrt(np.square(data[spine_column[i+1]]['x']-data[c]['x'])+np.square(data[spine_column[i+1]]['y']-data[c]['y']))
            dist_check = dist>p1
            mydata[c]["x"][dist_check]  = np.nan
            mydata[c]["y"][dist_check]  = np.nan
        if (i>1 and i<(len(spine_column)-1)):
            r_decision = False
            dist1=np.sqrt(np.square(data[spine_column[i-1]]['x']-data[c]['x'])+np.square(data[spine_column[i-1]]['y']-data[c]['y']))
            dist2=np.sqrt(np.square(data[spine_column[i+1]]['x']-data[c]['x'])+np.square(data[spine_column[i+1]]['y']-data[c]['y']))
            # further check the relative position:
            if i > 2:
                dist3 = np.sqrt(np.square(data["F_spine1"]['x']-data[c]['x'])+np.square(data["F_spine1"]['y']-data[c]['y']))
                if np.logical_or((dist3[0] > ((i-1)*p1+t1)),(dist3[0]<((i-3)*p1))):
                    r_decision = True
            dist_check= np.logical_or(((dist1>p1)|(dist2>p1)), r_decision)
            mydata[c]["x"][dist_check] = np.nan
            mydata[c]["y"][dist_check] = np.nan
        if i==(len(spine_column)-1):
            dist1=np.sqrt(np.square(data[spine_column[i-1]]['x']-data[c]['x'])+np.square(data[spine_column[i-1]]['y']-data[c]['y']))
            dist2=np.sqrt(np.square(data[spine_column[i-2]]['x']-data[c]['x'])+np.square(data[spine_column[i-2]]['y']-data[c]['y']))
            dist3 = np.sqrt(np.square(data["F_spine1"]['x']-data[c]['x'])+np.square(data["F_spine1"]['y']-data[c]['y']))
            r_decision = np.logical_or(dist3[0]>((i-1)*p1), dist3[0]<((i-4)*p1))
            dist_check=np.logical_or(((dist1>p1)|(dist2>p1)), r_decision)
            mydata[c]["x"][dist_check] = np.nan
            mydata[c]["y"][dist_check] = np.nan
        
    return mydata

# then filter and change the format of data to filled points
    
def spine_point(data):
    x_index = [0, 15, 18, 21, 24, 27, 30, 33]
    y_index = [1, 16, 19, 22, 25, 28, 31, 34]
    x = data.iloc[:,x_index]
    y = data.iloc[:,y_index]
    spline_point = np.column_stack([x,y])
    spline_point = spline_point.reshape((len(data),8,2), order = "F")

    return spline_point

def fill_tail(data):
    # data = filter_tail.dropna(subset = (("A_head","x"),('F_spine1','x')))
    data = spine_point(data)
    remain_list = []
    # first find the longest non_na in the point
    for i in range(data.shape[0]):
        not_na = np.unique(np.where(~np.isnan(data[i]))[0])
        if (len(not_na)>4):
            h = not_na[0]
            s1 = not_na[1]
            if (h == 0) & (s1==1):
                remain_list.append(i) ## filter  when do tail spine
                for j in range(len(not_na)):
                    if j > 1:
                        current = not_na[j]
                        pre = not_na[j-1]
                        point = current-pre
                        if point > 1:
                            dx = data[i][current][0]-data[i][pre][0]
                            dy = data[i][current][1]-data[i][pre][1]
                            for k in range(1, point):
                                data[i][pre+k][0] = data[i][pre][0]+k*dx/point
                                data[i][pre+k][1] = data[i][pre][1]+k*dy/point
    return(data[remain_list],remain_list)

## then rotate data


def rotation(data_overall, fill_data):
    data = data_overall
    n = data.shape[0]
    ## transfer spline data to point vector
    spline_point = fill_data

    # origin and reference vector
    head = np.column_stack([data.iloc[:,0],data.iloc[:,1]]) # dim = 216059, 2
    spline1 = np.column_stack([data.iloc[:,15],data.iloc[:,16]])
        # dim = 216059, 2
    head_r = head-spline1

    ##  rotation matrix 
    norm = np.zeros(len(head_r))
    for i in range(len(head_r)):norm[i] =  (np.linalg.norm(head_r[i]))
    #norm = np.array(norm)
    angle = np.column_stack([head_r[:,0]/norm, head_r[:,1]/norm])
    angle2 = np.column_stack([-angle[:,1],angle[:,0]])
    rot_matrix = np.column_stack([angle,angle2])

     ## rotate point coordinates
    spline_rotate = np.zeros((n,8,2))
    for i in range(n):
        x = np.zeros((8,2))
        k = 0
        for j in spline_point[i].reshape((8,2), order = "F"):
            x[k] = np.dot(rot_matrix[i].reshape(2,2),j-spline1[i])
            k = k+1
        spline_rotate[i] = x
    return(spline_rotate)


def tail_spline(rotate_points):
    tail = np.zeros((rotate_points.shape[0],8))
    j=0
    for i in range(len(rotate_points)):
        pts = rotate_points[i]
        pts = np.delete(pts,np.where(np.isnan(pts))[0],axis=0)
        k = pts.shape[0]
        tck, u = interpolate.splprep(pts.T, u=None, s=0.0) 
        yder = interpolate.splev(u, tck, der=1)
        z = np.full(8, np.nan)
        z[0:k] = yder[1]/yder[0]
        tail[i] = np.array(z)
    return(tail)
    
def plot_tail(beating, focus_region = range(10000,20000)):
    #data_focus = rotation(data.iloc[focus_region,:])
    #beating = tail_spline(data_focus)
    beating_df = pd.DataFrame(beating).dropna()
    fig,ax = plt.subplots(2,4,figsize=(20, 10))
    for j in range(8):
        xmin = np.quantile(beating_df.iloc[:,j],0.05)
        xmax = np.quantile(beating_df.iloc[:,j],0.95)
        ax[j//4,j%4].hist(beating_df.iloc[:,j], bins = 40, range = [xmin,xmax])
        ax[j//4,j%4].set_title("spine"+str(j))


#def rotation(data_overall, fill_data): ## data_overall should be filter_data.iloc[remain,:](dataframe), fill_na should be filled-array
#    data = data_overall
#    n = data.shape[0]
#    ## transfer spline data to point vector
#    spline_point = fill_data
#
#    # origin and reference vector
#    head = np.column_stack([data.iloc[:,0],data.iloc[:,1]]) # dim = 216059, 2
#    spline1 = np.column_stack([data.iloc[:,15],data.iloc[:,16]])
#        # dim = 216059, 2
#    head_r = head-spline1
#
#    ##  rotation matrix 
#    norm = np.zeros(len(head_r))
#    for i in range(len(head_r)):norm[i] =  (np.linalg.norm(head_r[i]))
#    #norm = np.array(norm)
#    angle = np.column_stack([head_r[:,0]/norm, head_r[:,1]/norm])
#    angle2 = np.column_stack([-angle[:,1],angle[:,0]])
#    rot_matrix = np.column_stack([angle,angle2])
#
#     ## rotate point coordinates
#    spline_rotate = np.zeros((n,8,2))
#    for i in range(n):
#        x = np.zeros((8,2))
#        k = 0
#        for j in spline_point[i].reshape((8,2), order = "F"):
#            x[k] = np.dot(rot_matrix[i].reshape(2,2),j-spline1[i])
#            k = k+1
#        spline_rotate[i] = x
#    return(spline_rotate)
#    
#def tail_spline(rotate_points,r = 15, t1 = 10, t2 = 5):
#    tail = np.zeros((rotate_points.shape[0],8))
#    j=0
#    nan_list = []
#    for i in range(len(rotate_points)):
#        pts = np.array(rotate_points[i])
#        index = [0]
#        omit = []
#        for k in range(len(pts)):
#            radius = np.linalg.norm(pts[k])
#            if radius<=(r*(k-1)+t1) and radius >(r*(k-2)-t2):
#                index.append(k)
#            else:
#                omit.append(k)
#        if len(index)<4:
#            nan_list.append(i)
#            tail[i] = np.full([1,8],np.nan)
#            continue
#            
#        pts_n = pts[index]
#        f = pts[max(index)][1]/pts[max(index)][0]
#        tck, u = interpolate.splprep(pts_n.T, u=None, s=0.0) 
#        yder = interpolate.splev(u, tck, der=1)
#        z = np.zeros(8)
#        z[index] = (yder[1]/yder[0])
#        z[omit[1:]] = f
#        tail[i] = np.array(z)
#    return(tail)
    
    

#def rotation(data):
#    n = data.shape[0]
#    ## transfer spline data to point vector
#    spline_point = []
#    x_index = [0, 15, 18, 21, 24, 27, 30, 33]
#    y_index = [1, 16, 19, 22, 25, 28, 31, 34]
#    x = data.iloc[:,x_index]
#    y = data.iloc[:,y_index]
#    spline_point = np.column_stack([x,y])
#    
#    # origin and reference vector
#    head = np.column_stack([data.iloc[:,0],data.iloc[:,1]]) # dim = 216059, 2
#    spline1 = np.column_stack([data.iloc[:,15],data.iloc[:,16]])
#        # dim = 216059, 2
#    head_r = head-spline1
#    
#    ##  rotation matrix 
#    norm = np.zeros(len(head_r))
#    for i in range(len(head_r)):norm[i] =  (np.linalg.norm(head_r[i]))
#    #norm = np.array(norm)
#    angle = np.column_stack([head_r[:,0]/norm, head_r[:,1]/norm])
#    angle2 = np.column_stack([-angle[:,1],angle[:,0]])
#    rot_matrix = np.column_stack([angle,angle2])
#    
#     ## rotate point coordinates
#    spline_rotate = np.zeros((n,8,2))
#    for i in range(n):
#        x = np.zeros((8,2))
#        k = 0
#        for j in spline_point[i].reshape((8,2), order = "F"):
#            x[k] = np.dot(rot_matrix[i].reshape(2,2),j-spline1[i])
#            k = k+1
#        spline_rotate[i] = x
#    
#    return(spline_rotate) 
#    
    

#        
    
#def tail_spline(rotate_points,r = 15, t1 = 10, t2 = 5):
#    tail = np.zeros((rotate_points.shape[0],8))
#    j=0
#    nan_list = []
#    for i in range(len(rotate_points)):
#        pts = np.array(rotate_points[i])
#        index = [0]
#        omit = []
#        for k in range(len(pts)):
#            radius = np.linalg.norm(pts[k])
#            if radius<=(r*(k-1)+t1) and radius >(r*(k-2)-t2):
#                index.append(k)
#            else:
#                omit.append(k)
#        if len(index)<4:
#            nan_list.append(i)
#            tail[i] = np.full([1,8],np.nan)
#            continue
#            
#        pts_n = pts[index]
#        f = pts[max(index)][1]/pts[max(index)][0]
#        tck, u = interpolate.splprep(pts_n.T, u=None, s=0.0) 
#        yder = interpolate.splev(u, tck, der=1)
#        z = np.zeros(8)
#        z[index] = (yder[1]/yder[0])
#        z[omit[1:]] = f
#        tail[i] = np.array(z)
#    return(tail)
    

    

