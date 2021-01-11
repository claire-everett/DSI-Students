#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:17:26 2021

@author: ryan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions
from contour_utils import find_centroid
from tqdm import tqdm
import cv2
fps = 40
IMAGELENGTH = 500

def get_centroids(contours):
    centroids = []
    
    for contour in contours:
        contour = contour[:,:2] #remove the curve score
        contour = np.array(contour[:,None,:], dtype=np.int32)
        centroid = find_centroid(contour)
        centroids.append(centroid)
    
    #I'm keeping a notation here because I have two ways of getting a baseline, the first is using 
    #the centroid of contour - head, the ohter is middle-point 50 pixels away from head along contour - head
    #I'm not sure which is better
    '''
    for i,contour in enumerate(contours):
        contour = contour[:,:2] #remove the curve score
        N = len(contour)
        head_index = head_indexs[i]
        centroid = (contour[(head_index+50)%N]+contour[(head_index-50)%N])/2
        centroids.append(centroid)    
    '''
    
    centroids = np.array(centroids)
    return centroids    

def get_head_coords(contours, head_indexs):
    Xcoords=[]
    Ycoords=[]
    for i,head_index in enumerate(head_indexs):
        N = len(contours[i])
        Xcoords.append(contours[i][head_index%N,0])
        Ycoords.append(contours[i][head_index%N,1])
    Xcoords=np.array(Xcoords)
    Ycoords=np.array(Ycoords)
    return Xcoords, Ycoords

def compute_cos(x1, x2, y1, y2):
    return np.arccos((x1*x2 + y1*y2)/np.sqrt(x1**2 + y1**2)/np.sqrt(x2**2 + y2**2))/np.pi*180

def get_projected_speed(x, y, basex, basey, n_prev = 10):
    n_after_x = np.concatenate((np.repeat(np.nan,n_prev),x[:-n_prev]))
    n_after_y = np.concatenate((np.repeat(np.nan,n_prev),y[:-n_prev]))
    distx = n_after_x - x
    disty = n_after_y - y
    
    TotalDist = np.sqrt(distx**2 + disty**2)
    horizonal_dist = np.abs(distx*basex+disty*basey)/(np.sqrt(basex**2 + basey**2))
    
    vertical_dist = np.sqrt(TotalDist**2 - horizonal_dist**2)  
    
    horizonal_speed = horizonal_dist / (n_prev / fps)
    vertical_speed = vertical_dist / (n_prev / fps) 
    return horizonal_speed, vertical_speed     

def projected_speed(contours, head_indexs, n_prev = 10):
    #find_centroid
    centroids = get_centroids(contours)
    
    #calculate the vertical and horizonal speed with the baseline of head-centroid
    Xcoords, Ycoords = get_head_coords(contours, head_indexs)
    
    centroid_Xcoords, centroid_Ycoords = centroids[:,0], centroids[:,1]
    n_after_x = np.concatenate((np.repeat(np.nan,n_prev),Xcoords[:-n_prev]))
    n_after_y = np.concatenate((np.repeat(np.nan,n_prev),Ycoords[:-n_prev]))
    distx = n_after_x-Xcoords
    disty = n_after_y-Ycoords
    centroid_distx = Xcoords-centroid_Xcoords
    centroid_disty = Ycoords-centroid_Ycoords
    
    TotalDist = np.sqrt(distx**2 + disty**2)
    horizonal_dist = np.abs(distx*centroid_distx+disty*centroid_disty)/(np.sqrt(centroid_distx**2 + centroid_disty**2))
    
    vertical_dist = np.sqrt(TotalDist**2 - horizonal_dist**2)
    
    horizonal_speed = horizonal_dist / (n_prev / fps)
    vertical_speed = vertical_dist / (n_prev / fps) 
    return horizonal_speed, vertical_speed 

def Angle_Distance_between_fish(contours1, contours2, head_indexs1, head_indexs2):
    #count in the width of fish tank when modifying the right fish's X value
    for i, contour in enumerate(contours2):
        contours2[i][:, 0] = contours2[i][:, 0] + IMAGELENGTH
        
    centroids1 = get_centroids(contours1)
    centroids2 = get_centroids(contours2)
    headx1, heady1 = get_head_coords(contours1, head_indexs1)
    headx2, heady2 = get_head_coords(contours2, head_indexs2)
    
    #L fish's centroid to head vector
    head_cen_vec1_x = headx1 - centroids1[:,0] 
    head_cen_vec1_y = heady1 - centroids1[:,1]
    
    #R fish's centroid to head vector
    head_cen_vec2_x = headx2 - centroids2[:,0]
    head_cen_vec2_y = heady2 - centroids2[:,1]
    
    #centroid of R fish to centroid of L fish
    cen1_cen2_x, cen1_cen2_y = centroids1[:,0] - centroids2[:,0], centroids1[:,1] - centroids2[:,1]
    
    #head of R fish to head of L fish
    head1_head2_x, head1_head2_y = headx1 - headx2, heady1 - heady2
    
    #get distance between centroid
    cen_dist = np.sqrt(cen1_cen2_x**2 + cen1_cen2_y**2)
    
    #get distance between heads
    head_dist = np.sqrt(head1_head2_x**2 + head1_head2_y**2)
    
    #get LR angle
    LR_Angle = compute_cos(head_cen_vec1_x, cen1_cen2_x, head_cen_vec1_y, cen1_cen2_y)
    
    #get RL angle
    RL_Angle = compute_cos(head_cen_vec2_x, cen1_cen2_x, head_cen_vec2_y, cen1_cen2_y)
    
    #get angle between directions of two fish
    Dir_Angle = compute_cos(head_cen_vec1_x, head_cen_vec2_x, head_cen_vec1_y, head_cen_vec2_y)
    
    #get forward,lateral speed of L fish in direction of R fish
    LRLS, LRFS = get_projected_speed(headx1, heady1, head_cen_vec2_x, head_cen_vec2_y)
    
    #get forward,lateral speed of L fish in direction of R fish
    RLLS, RLFS = get_projected_speed(headx2, heady2, head_cen_vec1_x, head_cen_vec1_y)
    
    return pd.DataFrame(np.array([cen_dist, head_dist, LR_Angle, RL_Angle, Dir_Angle,
                        LRLS, LRFS, RLLS, RLFS]).T,
                        columns=["center_distance", "head_distance", "LR_Angle", "RL_Angle", "head_dir_angle"
                        ,"L_lateral_speed", "L_forward_speed", "R_lateral_speed"
                        , "R_forward_speed"])


import pickle
with open("data/VM3_VM4_5.1.1_L_70000_150000_curve_scores","rb") as fp:
    contours_L = pickle.load(fp)
    
with open("data/VM3_VM4_5.1.1_L_70000_150000_head_index","rb") as fp:
    head_indexs_L = pickle.load(fp)

with open("data/VM3_VM4_5.1.1_R_70000_150000_curve_scores","rb") as fp:
    contours_R = pickle.load(fp)
    
with open("data/VM3_VM4_5.1.1_R_70000_150000_head_index","rb") as fp:
    head_indexs_R = pickle.load(fp)

feature_between_fish = Angle_Distance_between_fish(contours_L, contours_R, head_indexs_L, head_indexs_R)
feature_between_fish.to_csv("data/VM3_VM4_5.1.1_acrossFish.csv",index=False)
