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

def rotation(data):
    n = data.shape[0]
    ## transfer spline data to point vector
    spline_point = []
    x_index = [0, 15, 18, 21, 24, 27, 30, 33]
    y_index = [1, 16, 19, 22, 25, 28, 31, 34]
    x = data.iloc[:,x_index]
    y = data.iloc[:,y_index]
    spline_point = np.column_stack([x,y])
    
    ## reference vector
    head = np.column_stack([data.iloc[:,0],data.iloc[:,1]]) # dim = 216059, 2
    spline1 = np.column_stack([data.iloc[:,15],data.iloc[:,16]])
    # dim = 216059, 2
    head_r = head-spline1 # reference vector to x axis
    
    ##  rotation matrix 
    norm = []
    for i in range(len(head_r)):
        norm.append(np.linalg.norm(head_r[i]))
    norm = np.array(norm)
    angle = np.column_stack([head_r[:,0]/norm, head_r[:,1]/norm])
    angle2 = np.column_stack([-angle[:,1],angle[:,0]])
    rot_matrix = np.column_stack([angle,angle2])
    
    ## rotate point coordinates
    spline_rotate = []
    for i in range(n):
        x = []
        for j in spline_point[i].reshape((8,2), order = "F"):
            x.append((np.dot(rot_matrix[i].reshape(2,2),j-spline1[i])))
        spline_rotate.append(x)
    
    return(spline_rotate)
    
    
def tail_spline(rotate_points,r = 15, t1 = 10, t2 = 5):
    tail = []
    j=0
    for i in range(len(rotate_points)):
        pts = np.array(rotate_points[i])
        index = [0]
        omit = []

        for k in range(len(pts)):
            radius = np.linalg.norm(pts[k])
            if radius<=(r*(k-1)+t1) and radius >(r*(k-2)-t2):
                index.append(k)
            else:
                omit.append(k)
        if len(index)<4:
            print(i)
            continue        
        pts_n = pts[index]
        f = pts[max(index)][1]/pts[max(index)][0]
        tck, u = interpolate.splprep(pts_n.T, u=None, s=0.0) 
        yder = interpolate.splev(u, tck, der=1)
        z = np.zeros(8)
        z[index] = (yder[1]/yder[0])
        z[omit[1:]] = f
        tail.append(z)
    tail = np.array(tail)
    return(tail)
        
    

