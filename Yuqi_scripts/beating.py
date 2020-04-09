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
    x_index = [i for i in range(15,15+7*3) if (i+1)%3 == 1]
    y_index = [i for i in range(15,15+7*3) if (i+1)%3 == 2]
    for j in range(n):
        x = data.iloc[j,x_index]
        y = data.iloc[j,y_index]
        spline_point.append(np.column_stack([x,y]))
    
    ## reference vector
    head = np.column_stack([data["A_head"]["x"],data["A_head"]["y"]]) # dim = 216059, 2
    spline1 = np.column_stack([data["F_spine1"]["x"],data["F_spine1"]["y"]])
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
        for j in spline_point[i]:
            x.append(np.dot(rot_matrix[1].reshape(2,2),j-spline1[i]))
        spline_rotate.append(x)
    
    return(spline_rotate)
    
    
def tail_spline(rotate_points):
    tail = []
    for i in range(len(rotate_points)):
        pts = np.array(rotate_points[i])
        tck, u = interpolate.splprep(pts.T, u=None, s=0.0, per=1)
        x = np.vstack(rotate_points[i])[:,0]
        yder = interpolate.splev(x, tck, der=1)
        tail.append(yder)
    return(tail)
        
    

