#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:31:06 2020

@author: Claire
"""

#%%
#Imports

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import interpolate
import seaborn as sns; sns.set()
from hmmlearn import hmm
# import ssm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook as tqdm
from sklearn.cluster import KMeans
from itertools import permutations
import moviepy
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%%
#Functions


## Loading Data
# Loading Data for one fish
def auto_scoring_tracefilter(data,p0=20,p2=15):
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum', 'C_tailbase', 'D_tailtip','E_leftoperculum']
    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
            mydata[b][j][xdiff_check] = np.nan
 
            origin_check = mydata[b][j] < p2
            mydata[origin_check] = np.nan

    return mydata

def getfiltereddata_1(h5_files):
    file_handle1 = h5_files[0]

    with pd.HDFStore(file_handle1,'r') as help1:
        data_auto1 = help1.get('df_with_missing')
        data_auto1.columns= data_auto1.columns.droplevel()
        data_auto1_filt = auto_scoring_tracefilter (data_auto1)
 
    
    return data_auto1_filt

#%%
# Loading Data for two fish

def getfiltereddata_2(h5_files):
    file_handle1 = h5_files[0]

    with pd.HDFStore(file_handle1,'r') as help1:
        data_auto1 = help1.get('df_with_missing')
        data_auto1.columns= data_auto1.columns.droplevel()
        data_auto1_filt = auto_scoring_tracefilter (data_auto1)
     
    file_handle2 = h5_files[1]

    with pd.HDFStore(file_handle2,'r') as help2:
        data_auto2 = help2.get('df_with_missing')
        data_auto2.columns= data_auto2.columns.droplevel()
        data_auto2_filt = auto_scoring_tracefilter(data_auto2)
        data_auto2_filt['A_head']['x'] = data_auto2_filt['A_head']['x'] + 500
        data_auto2_filt['B_rightoperculum']['x'] = data_auto2_filt['B_rightoperculum']['x'] + 500
        data_auto2_filt['E_leftoperculum']['x'] = data_auto2_filt['E_leftoperculum']['x'] + 500
    
    
    data_auto1_filt['zeroed','x'] = data_auto1_filt['A_head']['x'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[0]
    data_auto1_filt['zeroed','y'] = data_auto1_filt['A_head']['y'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[1]
    
    data_auto2_filt['zeroed','x'] = data_auto2_filt['A_head']['x'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[0]
    data_auto2_filt['zeroed','y'] = data_auto2_filt['A_head']['y'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[1]
    
    
    return data_auto1_filt, data_auto2_filt

#%%
## Basic Functions

def mydistance(pos_1,pos_2):
    '''
    Takes two position tuples in the form of (x,y) coordinates and returns the distance between two points
    '''
    x0,y0 = pos_1
    x1,y1 = pos_2
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)

    return dist


def nanarccos (floatfraction):
    con1 = ~np.isnan(floatfraction)
    
    if con1:
        cos = np.arccos(floatfraction)
        OPdeg = np.degrees(cos)
    
    else:
        OPdeg = np.nan
    
    return (OPdeg)

        
def vecnanarccos():
    
    A = np.frompyfunc(nanarccos, 1, 1)
    
    return A
  
    
def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    
    return (data['x'],data['y'])


def lawofcosines(line_1,line_2,line_3):
    '''
    Takes 3 series, and finds the angle made by line 1 and 2 using the law of cosine
    '''
 
    num = line_1**2 + line_2**2 - line_3**2
    denom = (line_1*line_2)*2
    floatnum = num.astype(float)
    floatdenom = denom.astype(float)
    floatfraction = floatnum/floatdenom
    OPdeg = vecnanarccos()(floatfraction)
    
    return OPdeg

#%%
    
# Defining Operculum
    
def auto_scoring_get_opdeg(data_auto):
    '''
    Function to automatically score operculum as open or closed based on threshold parameters. 
    
    Parameters: 
    data_auto: traces of behavior collected as a pandas array. 
    thresh_param0: lower threshold for operculum angle
    thresh_param1: upper threshold for operculum angle
    
    Returns:
    pandas array: binary array of open/closed scoring
    '''
    # First collect all parts of interest:
    poi = ['A_head','B_rightoperculum','E_leftoperculum']
    HROP = mydistance(coords(data_auto[poi[0]]),coords(data_auto[poi[1]]))
    HLOP = mydistance(coords(data_auto[poi[0]]),coords(data_auto[poi[2]]))
    RLOP = mydistance(coords(data_auto[poi[1]]),coords(data_auto[poi[2]]))
    
    Operangle = lawofcosines(HROP,HLOP,RLOP)
    
    return Operangle


def triangle_shape(data_auto):
    '''
    Function to automatically score operculum triangle as acute or obtuse  based on ratio of angles
    
    Parameters:
        data_auto: traces of behavior collected as a pandas array
    
    Returns:
    pandas array: binary array of open/closed scoring
    '''
    
    poi = ['A_head','B_rightoperculum','E_leftoperculum']
    
    HROP = mydistance(coords(data_auto[poi[0]]),coords(data_auto[poi[1]]))
    HLOP = mydistance(coords(data_auto[poi[0]]),coords(data_auto[poi[2]]))
    RLOP = mydistance(coords(data_auto[poi[1]]),coords(data_auto[poi[2]]))
    
    HeadAngle = lawofcosines(HROP,HLOP,RLOP)
    LOpAngle =  lawofcosines(HROP,RLOP, HLOP)
    ROpAngle =  lawofcosines(HLOP, RLOP, HROP)
    
    Ratio = HeadAngle/LOpAngle/ROpAngle
    
    return Ratio

def oper_speed(Operangle,fps=40):
    '''
    Function that finds the derivative of the operculum angle to track rate of change 
    
    Parameters:
    Operangle = the degree of angle over time
    fps = frame rate of recording
    
    Returns:
    pandas array: Operculum speed over time  
    '''
    movement= np.diff(Operangle)
    OpSpeed=abs(movement/(1/fps))
    OpSpeed = np.insert(OpSpeed, len(OpSpeed), np.nan, axis = 0)
    
    return OpSpeed

def oper_max(Operangle,width=20):
    '''
    Function that calculates the max operculum angle for the x frames surrounding any given point (ie. 20 before/20after)
    
    Parameters: 
    Operangle = the degree of angle over time
    width = the range you're searching over for local max
    '''
    
    OpMax=[]
    for i in range(len(Operangle)):
        local=Operangle[max(0,i-width):min(len(Operangle)-1,i+width)]
        OpMax.append(local.max())
        
    return OpMax
#%%

# Measuring Tail Beating Behavior
def conditional_orientation(data, orientation, threshold):
    '''
    creates a dataframe of only frames in which the fish
    is turned inside a certain orientation.
    '''
    boolean = orientation.apply(lambda x: 1 if x > threshold else 0) 
    df = pd.DataFrame(data, index = boolean)
    df = df.loc[1]
    
    return df


#%%
    
# Yuqi's codes, does the rotation first, then in one function filters and calculates the derivatives. Does not compute a 
    # curvature
    
# My ideas to make roattion faster: do the filtering first, this will greatly reduce the 
    # number of data you would need to rotate 
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
    norm = np.linalg.norm(head_r)
#    for i in range(len(head_r)):
#        norm.append(np.linalg.norm(head_r[i]))
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
            tail.append(np.nan)
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
    
#%%


# filtering with yuyang's method but need to play around with the distance that is best and creates the most accurate
# spline
    
def Combine_filter (data, p0 = 20  , p1 = 20):
    '''
    returns a whole dataframe with NANs according to general and specific spine criterion
    '''
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum','E_leftoperculum',"F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7"]
    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
            mydata[b][j][xdiff_check] = np.nan
    spine_column=["A_head", "F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7",'D_tailtip','C_tailbase']
    for i,c in enumerate(spine_column):
        if (i>1 and i<(len(spine_column)-1)):
            dist1=np.sqrt(np.square(mydata[spine_column[i-1]]['x']-mydata[c]['x'])+np.square(mydata[spine_column[i-1]]['y']-mydata[c]['y']))
            dist2=np.sqrt(np.square(mydata[spine_column[i+1]]['x']-mydata[c]['x'])+np.square(mydata[spine_column[i+1]]['y']-mydata[c]['y']))
            dist_check=np.logical_and(dist1>p1,dist2>p1)
            mydata[c]["x"][dist_check] = np.nan
            mydata[c]["y"][dist_check] = np.nan
        if i==(len(spine_column)-1):
            dist1=np.sqrt(np.square(mydata[spine_column[i-1]]['x']-mydata[c]['x'])+np.square(mydata[spine_column[i-1]]['y']-mydata[c]['y']))
            dist2=np.sqrt(np.square(mydata[spine_column[i-2]]['x']-mydata[c]['x'])+np.square(mydata[spine_column[i-2]]['y']-mydata[c]['y']))
            dist_check=np.logical_and(dist1<p1,dist2<p1)
            mydata[c]["x"][dist_check] = np.nan
            mydata[c]["y"][dist_check] = np.nan
    # FILLS IN THE NANS
    # mydata = mydata.fillna(method = 'ffill')
    return(mydata)
#%%
    
## this I don't think is actually working!? I can't tell... need help. However I think the ideas
## are there, as described in the function.
def Combine_filter_CE (data, p0 = 20  , p1 = 5):
    '''
    
    This function filters  based on 1) jumps between frames, 2) the liklihood of the position, 3) the closeness of the points
    It does not filter based on any distance (Yuqi and Yuyang can perfect)
    
    returns a whole dataframe with NANs according to general and specific spine criterion
    '''
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum','E_leftoperculum',"F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7", 'D_tailtip','C_tailbase']
    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
            mydata[b][j][xdiff_check] = np.nan
        threshold = mydata[b]['likelihood'].quantile(.5)
        lik_check = mydata[b]['likelihood'] > threshold
        mydata[b]['likelihood'][lik_check] = np.nan
    spine_column=["A_head", "F_spine1","G_spine2","H_spine3","I_spine4","J_spine5","K_spine6","L_spine7",'D_tailtip','C_tailbase']
    perm = permutations(spine_column, 2)
    for i in list(perm):
        rel_dist = mydistance(coords(mydata[i[0]]),coords(mydata[i[1]]))
        print(mydata[i[0]])
        rel_dist_check = rel_dist < p1
        mydata[rel_dist_check] = np.nan 
    return(mydata) 
#%%
    
## This block of code makes a video of the points for a certain duration of time. Duration = 10 means 10 seconds
## of video. Fps is the frames per second. The # of frames you are using to make the video should be at least duration
## times 40. The name of the dataframe should be entered into the function, at least for now.
## make sure to change the name of the output video at least until everything is functionalized.
    
duration = 100
fps = 40
fig, ax = plt.subplots()

def make_frame(time):
    timeint = int(time*fps)
    ax.clear()
    x = data_auto2_filt['A_head']['x'][timeint]
    y = data_auto2_filt['A_head']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['F_spine1']['x'][timeint]
    y = data_auto2_filt['F_spine1']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['G_spine2']['x'][timeint]
    y = data_auto2_filt['G_spine2']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['H_spine3']['x'][timeint]
    y = data_auto2_filt['H_spine3']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['I_spine4']['x'][timeint]
    y = data_auto2_filt['I_spine4']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['I_spine4']['x'][timeint]
    y = data_auto2_filt['I_spine4']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['J_spine5']['x'][timeint]
    y = data_auto2_filt['J_spine5']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['K_spine6']['x'][timeint]
    y = data_auto2_filt['K_spine6']['y'][timeint]
    ax.plot(x,y,'o')
    x = data_auto2_filt['L_spine7']['x'][timeint]
    y = data_auto2_filt['L_spine7']['y'][timeint]
    ax.plot(x,y,'o')
    
    ax.set_ylim([0,500])
    ax.set_xlim([0,500])
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)
animation.write_videofile("unfilt_animation_100.mp4", fps=40)

#%%

## change so gives out just the points dataframe? (is it a dataframe?)
pts_all = []
x_all = []
y_all = []
def Combine_pts(data):
    for i in np.arange(len(data)):
        test_point = data.iloc[i,:]
        x = test_point[[0, 15, 18, 21, 24, 27, 30, 33]]
        y = test_point[[1, 16, 19, 22, 25, 28, 31, 34]]
        pts=np.vstack([x,y]).T
        pts=pts[~np.isnan(pts).any(axis=1)]
        pts_all.append(pts)
        print()
        if(pts.shape[0]>=4):
            tck,u=interpolate.splprep(pts.T, u=None, s=0.0)
            u_new = np.linspace(u.min(), u.max(), 100)
            x_new, y_new = interpolate.splev(u_new, tck, der=0)
            x_all.append(x_new)
            y_all.append(y_new)
    return( x_all, y_all)
#%%
x_all, y_all = Combine_pts(Combine_filter_CE(data_auto2_filt))
duration = 10
fps = 40
fig, ax = plt.subplots()

def make_frame(time):
    timeint = int(time*fps)
    ax.clear()
    ax.plot(x_all[timeint], y_all[timeint], 'b--')
    ax.set_ylim([0,500])
    ax.set_xlim([0,500])
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)
animation.write_videofile("filt_test.mp4", fps=40)
#%%
def Combine_rotate(data):
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
    norm = np.linalg.norm(head_r)
#    for i in range(len(head_r)):
#        norm.append(np.linalg.norm(head_r[i]))
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

def Combine_curv(data):
    curvature=[]
    spline_info=[]
    for i in np.arange(len(data)):
        test_point = data.iloc[i,:]
        x = test_point[[0, 15, 18, 21, 24, 27, 30, 33]]
        y = test_point[[1, 16, 19, 22, 25, 28, 31, 34]]
        pts=np.vstack([x,y]).T
        pts=pts[~np.isnan(pts).any(axis=1)]
        if(pts.shape[0]>=4):
            tck,u=interpolate.splprep(pts.T, u=None, s=0.0)
            spline_info.append([tck,u])
            dx1,dy1=interpolate.splev(u,tck,der=1)#why is this only a first order derivative?
            dx2,dy2=interpolate.splev(u,tck,der=2)# and this a second order?
            k=(dx1*dy2-dy1*dx2)/np.power((np.square(dx1)+np.square(dy1)),3/2)
            max_k=abs(k).max()
            curvature.append(max_k)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = interpolate.splev(u_new, tck, der=0)
            plt.figure()
            plt.plot(pts[:,0], pts[:,1], 'ro')
            plt.plot(x_new, y_new, 'b--')
            plt.show()
        else:
            curvature.append(np.nan)
            spline_info.append(np.nan)
    return(curvature)
    


#%%
# Measuring Gaze and Orientation

def midpoint (pos_1, pos_2, pos_3, pos_4):
    '''
    give definition pos_1: x-value object 1, pos_2: y-value object 1, pos_3: x-value object 2
    pos_4: y-value object 2
    '''
    midpointx = (pos_1 + pos_3)/2
    midpointy = (pos_2 + pos_4)/2

    return (midpointx, midpointy)
    
def orientation2(data_auto):
    ## take out the head coordinate
    head_x = data_auto["A_head"]["x"]
    head_y = data_auto["A_head"]["y"]
    
    ## get the midpoint coordinate
    mid_x = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[0]
    mid_y = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[1]
    
    ## calculate cos theta
    cos_angle = list()
    for i  in range(data_auto.shape[0]):
        inner_product = (head_x[i]-mid_x[i])*head_x[i]
        len_product = ((head_x[i]-mid_x[i])**2+((head_y[i]-mid_y[i])**2))**(0.5)*head_x[i]
        cos_angle.append(inner_product/len_product)
    
    cos_angle = np.array(cos_angle)
    
    return cos_angle

def heading_angle(data_auto_arg):
    '''
    Function looks at orientation of the fish across a trial. It takes in teh dataframe and returns the 
    orientation for each frame. A degree of East = 0, North = 90, West = 180, South = 270
    '''
    # First collect all parts of interest:
    poi = ['zeroed']
    origin = pd.DataFrame(0.,index = data_auto_arg[poi[0]]['x'].index, columns = ['x','y'])
    distone = pd.Series(1, index = data_auto_arg[poi[0]]['x'].index)
    plusx = origin['x'] + 1
    plusy = origin['y']
    HO = mydistance(coords(data_auto_arg[poi[0]]), coords(origin))
    OP = distone
    PH  = mydistance(coords(data_auto_arg[poi[0]]),(plusx, plusy))

    
    out = lawofcosines(HO, OP, PH)
    return out

def gaze_tracking(fish1,fish2):
    """
    A function that takes in two dataframes corresponding to the two fish. Assymetric. Fish one is the one gazing, fish two is being gazed at. 
    
    Parameters: 
    Fish1: A dataframe of tracked points. 
    Fish2: A dataframe of tracked points. 
    Output:
    A vector of angles between 0 and 180 degrees (180 corresponds to directed gaze).
    """

    ## Get the midpoint of the gazing fish. 
    midx,midy = midpoint(fish1['B_rightoperculum']['x'], fish1['B_rightoperculum']['y'], fish1['E_leftoperculum']['x'], fish1['E_leftoperculum']['y'] )
    line1 = mydistance((midx, midy), (fish1['A_head']['x'], fish1['A_head']['y']))
    line2 = mydistance((fish1['A_head']['x'], fish1['A_head']['y']), (fish2['A_head']['x'], fish2['A_head']['y']))
    line3 = mydistance((fish2['A_head']['x'], fish2['A_head']['y']), (midx, midy))

    angle = lawofcosines(line1, line2, line3)
    return angle

#%%
# Measuring Speed and Angle

def speed (data, fps = 40):
    
    ''' function that calculates velocity of x/y coordinates
    plug in the xcords, ycords, relevant dataframe, fps
    return the velocity as column in relevant dataframe'''
    poi = ['A_head']
    (Xcoords, Ycoords)= coords(data[poi[0]])
    distx = Xcoords.diff() 
    disty = Ycoords.diff()
    TotalDist = np.sqrt(distx**2 + disty**2)
    Speed = TotalDist / (1/fps) #converts to seconds
    return Speed

def turning_angle(data_auto):
    head_x = data_auto["A_head"]["x"]
    head_y = data_auto["A_head"]["y"]
    
    ## get the midpoint coordinate
    mid_x = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[0]
    mid_y = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[1]
    ##find the 
    cur_vec=np.vstack((head_x-mid_x,head_y-mid_y)).T
    prev_vec=np.vstack((np.append((head_x-mid_x)[40:],np.repeat(np.nan,40)),np.append((head_y-mid_y)[40:],np.repeat(np.nan,40)))).T
    inner_product=np.sum(np.multiply(cur_vec,prev_vec),axis=1)
    cur_norm=np.sum(np.multiply(cur_vec,cur_vec),axis=1)
    prev_norm=np.sum(np.multiply(prev_vec,prev_vec),axis=1)
    cos=inner_product/np.sqrt(cur_norm*prev_norm)
    angle=np.arccos(cos)
    return angle/np.pi*180

#%%

#Manual Scoring
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

## Manual Scoring
excel_files = glob(os.path.join('.', '*.xlsx'))
#    
file_handle1 = excel_files[2]
data_manual1 = pd.read_excel(file_handle1)


#%%
# Loading Data
    
home_dir = '.'
h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
print(h5_files)

## Packaged up some of the upload code. 
data_auto1_filt,data_auto2_filt = getfiltereddata_2(h5_files) 

#%%

data_auto2_filt = data_auto1_filt[:1000]

#%%
data_auto2_filt['zeroed','x'] = data_auto2_filt['A_head']['x'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[0]
data_auto2_filt['zeroed','y'] = data_auto2_filt['A_head']['y'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[1]
    
Test = pd.DataFrame()
Test['orientation'] = (heading_angle(data_auto2_filt))
# Test['curve'] = Combine_curv(Combine_filter(data_auto2_filt))
Test['operculum']= np.array(auto_scoring_get_opdeg(data_auto2_filt), dtype = "float64")
# Test['speed'] = speed(data_auto2_filt)


#%%
Test = Test.fillna(method = 'bfill')

x = StandardScaler().fit_transform(Test)

pca = PCA()
pca.fit(x)
pcs=pca.transform(x)


pcs=pcs[:,:2]
pcs = np.clip(pcs,-3,3)
plt.scatter(pcs[:,0],pcs[:,1],s = 1)
#%%

## kmeans clustering

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10)
kmeans.fit(pcs)

y=kmeans.predict(pcs)
plt.scatter(pcs[:, 0], pcs[:, 1], c=y,s=1, cmap='viridis')
#%%

# using HMM to define clusters
hmm_3=hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
hmm_3.fit(pcs)
y=hmm_3.predict(pcs)
plt.scatter(pcs[:, 0], pcs[:, 1], c=y,s=1, cmap='viridis')
#%%

# Creating a PCA & Performing kmeans

## Defining Operculum Features

Oper_Angle = np.array(auto_scoring_get_opdeg(data_auto1_filt), dtype = "float64")
Oper_Max = np.array(oper_max(Oper_Angle), dtype = "float64")
Oper_Speed = np.array(oper_speed(Oper_Angle), dtype = "float64")
Oper_Triangle = np.array(triangle_shape(data_auto1_filt), dtype = "float64")

## Binning 
Mult = int(len(data_auto1_filt)/40)
index = list(range(40))* Mult
row = np.repeat(list(range(Mult)),40)

#%%
## Creating DataFrame
#data1 = pd.DataFrame(data =[Oper_Angle, Oper_Max, Oper_Speed, Oper_Triangle]).T
data1 = pd.DataFrame(data =[Oper_Angle, Oper_Triangle, Oper_Max]).T

#data1.columns = ['Oper_Angle', 'Oper_Max', 'Oper_Speed', 'Oper_Triangle']
data1.columns = ['Oper_Angle', 'Oper_Triangle', 'Oper_Max']


#%%

data = pd.DataFrame(data1).dropna() 
data = data[88660:98660]

data2 = data1.fillna(method="ffill")
data2 = data2[88660:98660]


x = StandardScaler().fit_transform(data2)


pca = PCA()
pca.fit(x)
pcs=pca.transform(x)


pcs=pcs[:,:2]
plt.scatter(pcs[:,1],pcs[:,0],s=1)

#%%

## kmeans clustering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10)
kmeans.fit(pcs)

y=kmeans.predict(pcs)
plt.scatter(pcs[:, 1], pcs[:, 0], c=y,s=3, cmap='viridis')
#%%

# Color the PCA Map based on Manual scoring 
# Add the Binary Feature onto the data2 and use that column to color, compare to the kmeans

Manual1 = manual_scoring(data_manual1, data2)
Manual1[10000] = 0

#%%
## Attempting Reorganizing, to bin the data 
#def reorg(data1, Column):
#    
#    '''
#    Input: pd.DataFrame with columns and the name of desired column (string) 
#    '''
#    Output = pd.DataFrame(Column)
#    Output['index'] = data1['index']
#    Output['row'] = data1['row']
#    Output.columns = ['Column', 'index', 'row']
#    
#    return Output
#
##def re_column(df_column):
##    df_column = df.column
##    Output = Output.pivot(index = Output['row2'], columns = Output['index2'], values = Output[Output.columns[0]])
##    Output.columns = (map(add,str(Column)*40,list(map(str,range(40)))))
##    return Output
#    
#df_Angle = reorg(data1, data1['Oper_Angle'])
#df_Max = reorg(data1, data1['Oper_Max'])
#df_Speed = reorg(data1, data1['Oper_Speed'])
#df_Triangle = reorg(data1, data1['Oper_Triangle'])
#
#df_Angle = df_Angle.pivot(index = df_Angle['row'], columns = df_Angle['index'], values = df_Angle['Column'])

#  
#data = pd.DataFrame(data1).dropna()  
### Binning Data
#
#''' Skipping for now '''
#
### Standardizing Features
#
#from sklearn.preprocessing import StandardScaler
#x = StandardScaler().fit_transform(data)

## 




#%%    
# Creating an HMM

## feature:
data_auto1_filt['zeroed','x'] = data_auto1_filt['A_head']['x'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[0]

data_auto1_filt['zeroed','y'] = data_auto1_filt['A_head']['y'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[1]
xx = np.array(turning_angle(data_auto1_filt), dtype = "float64")
yy = np.array(auto_scoring_get_opdeg(data_auto1_filt), dtype = "float64")
zz = np.array(speed(data_auto1_filt), dtype = "float64")
data1 = np.column_stack((xx,yy,zz))
data1 = data1[90094:100094]
data = pd.DataFrame(data1).dropna()


## feature exploration
plt.hist(xx) # right skewed, log transformation?
plt.hist(yy)

plt.boxplot

#%%
## setting hmm
# Set the parameters of the HMM
T = 8484   # number of time bins
K = 3       # number of discrete states
D = 3       # data dimension

## use EM to infer the model
data_em = np.array(data)
N_iters = 50
K = 3
hmm = ssm.HMM(K, D, observations="gaussian")
hmm_lls = hmm.fit(data_em, method="em", num_em_iters=N_iters)
inferred_states = hmm.most_likely_states(data_em)

## plot EM results
plt.plot(hmm_lls, label="EM")
# plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")

## plot the inferred states
plt.figure(figsize=(8, 4))
plt.imshow(inferred_states[None, :], aspect="auto",cmap = "gray")
plt.xlim(0, T)
plt.ylabel("inferred\nstate")
plt.yticks([])
plt.title("states plot using peri and turning angle and speed")
plt.show()

## transition matrix
learned_transition_mat = hmm.transitions.transition_matrix

fig = plt.figure(figsize=(8, 4))

im = plt.imshow(learned_transition_mat, cmap='gray')
plt.title("Learned Transition Matrix using peri and orientation")

cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

#%%

## Makes Transition Plot
matplotlib.rc_file_defaults()

means = pd.DataFrame(hmm.observations.mus)


F = hmm.transitions.transition_matrix

B = F[~np.eye(F.shape[0],dtype=bool)].reshape(F.shape[0],-1)
B = np.copy(F)
B[np.where(np.eye(B.shape[0]))] = np.nan

im = plt.imshow(B, cmap='gray')

plt.savefig("transitionmatrix.pdf")
#%%

## visualizeing emission matrix

sigmas = hmm.observations.Sigmas

variance = []
for i in np.arange(3):
    state = sigmas[i]
    var = np.diag(state)
    variance.append(var)
StateSD = np.sqrt(np.stack(variance,axis = 0)).T
fig,ax = plt.subplots()
colors = ["cyan","purple","orange"]
coords = ["turningangle","HeadX","HeadY"]
for mi,m in enumerate(means.values.T):
    print(m)
    sd = StateSD[mi]
    ax.plot(m,color = colors[mi],label = coords[mi])
    ax.plot(m-sd,color = colors[mi],linestyle = '--')
    ax.plot(m+sd,color = colors[mi],linestyle = '--')

#plt.plot(means[means.columns[:3]], "b")
#plt.plot(means[means.columns[:3]]-StateSD,"--")
#plt.plot(means[means.columns[:3]]+StateSD,"--")
ax.set_xticklabels(['blue', 'red', 'yellow'])
ax.set_xticks(range(4))
plt.legend(loc = 0)