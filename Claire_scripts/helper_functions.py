#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:31:45 2020

@author: Claire
"""

#%%

# Loading python packages
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate 
import matplotlib
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns; sns.set()
from scipy.signal import argrelextrema

#%%

# Basic functions for automated data

def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    
    return (data['x'],data['y'])

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
#    
    return Speed

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

def midpoint (pos_1, pos_2, pos_3, pos_4):
    '''
    give definition pos_1: x-value object 1, pos_2: y-value object 1, pos_3: x-value object 2
    pos_4: y-value object 2
    '''
    midpointx = (pos_1 + pos_3)/2
    midpointy = (pos_2 + pos_4)/2

    return (midpointx, midpointy)

def orientation(data_auto):
    ## take out the head coordinate
    head_x = data_auto["A_head"]["x"]
    head_y = data_auto["A_head"]["y"]
    
    ## get the midpoint coordinate
    mid_x = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[0]
    mid_y = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[1]
    
    ## calculate cos theta
    head_ori=np.array([head_x-mid_x,head_y-mid_y]).T
    ref=np.array([1,0])
    inner_product =head_ori.dot(ref)
    cos=inner_product/np.sqrt(np.sum(np.multiply(head_ori,head_ori),axis=1))
    angle=np.arccos(cos)/np.pi*180
    det=head_ori[:,1]>0
    angle[~det]=angle[~det]+180
    angle=np.minimum(angle,360-angle)
    return angle

def periculum_speed(periculum,fps=40):
    periculum=periculum.fillna(method="bfill")
    movement=np.diff(periculum)
    movement=np.insert(movement,0,np.nan)
    p_speed=abs(movement/(1/fps))
    return p_speed
''' fit a sine curve for the periculum data'''
from scipy import optimize
def sin_curve(x, b,c,d):
    return  30*np.sin(b * x+c)+d
def find_params(periculum,start=100000,end=100400):
    params, params_covariance = optimize.curve_fit(sin_curve, range(start,end), periculum[start:end],p0=[0.05,0,70])
    return params

def find_error(periculum):
    params=find_params(periculum)
    predict=sin_curve(range(len(periculum)),params[0],params[1],params[2])
    error=predict-periculum
    return abs(error)

def turning_angle(data_auto):
    head_x = data_auto["A_head"]["x"]
    head_y = data_auto["A_head"]["y"]
    
    ## get the midpoint coordinate
    mid_x = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[0]
    mid_y = midpoint(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])[1]
    ##find the 
    cur_vec=np.vstack((head_x-mid_x,head_y-mid_y)).T
    prev_vec=np.vstack((np.repeat(np.nan,40),np.append((head_x-mid_x)[:-40]),np.append(np.repeat(np.nan,40),(head_y-mid_y)[40:]))).T
    inner_product=np.sum(np.multiply(cur_vec,prev_vec),axis=1)
    cur_norm=np.sum(np.multiply(cur_vec,cur_vec),axis=1)
    prev_norm=np.sum(np.multiply(prev_vec,prev_vec),axis=1)
    cos=inner_product/np.sqrt(cur_norm*prev_norm)
    angle=np.arccos(cos)
    return angle/np.pi*180
def turning_angle_spine(data_auto):
    spine1_x = data_auto["F_spine1"]["x"]
    spine1_y = data_auto["F_spine1"]["y"]
    
    if "mid_spine1_spine2" in data_auto.columns:
        spine1_5_x=data_auto["mid_spine1_spine2"]["x"]
        spine1_5_y=data_auto["mid_spine1_spine2"]["y"]
    else:
        spine1_5_x=data_auto["G_spine2"]["x"]
        spine1_5_y=data_auto["G_spine2"]["y"]
    ##find the direction in the previous second
    cur_vec=np.vstack((spine1_x-spine1_5_x,spine1_y-spine1_5_y)).T
    prev_vec=np.vstack((np.append(np.repeat(np.nan,40),(spine1_x-spine1_5_x)[:-40]),np.append(np.repeat(np.nan,40),(spine1_y-spine1_5_y)[:-40]))).T
    inner_product=np.sum(np.multiply(cur_vec,prev_vec),axis=1)
    cur_norm=np.sum(np.multiply(cur_vec,cur_vec),axis=1)
    prev_norm=np.sum(np.multiply(prev_vec,prev_vec),axis=1)
    cos=inner_product/np.sqrt(cur_norm*prev_norm)
    angle=np.arccos(cos)
    return angle/np.pi*180

def find_local_max_per(periculum,width=20):
    '''use for loop, tedious'''
    re=[]
    for i in range(len(periculum)):
        local=periculum[max(0,i-width):min(len(periculum)-1,i+width)]
        re.append(local.max())
    return re

def compute_cos(x,y):
    #x,y is a (n,2) vector, output a nx1 vector of cosine for each row
    inner_product=np.sum(np.multiply(x,y),axis=1)
    norm_x=np.sum(np.multiply(x,x),axis=1)
    norm_y=np.sum(np.multiply(y,y),axis=1)
    inner_product/np.sqrt(norm_x*norm_y)


def myvelocity (xcoords, ycoords):
    xdiff = []
    for i,value in enumerate (xcoords):
        if i < len(xcoords)-1:
            x1 = xcoords.iloc[i]
            x2 = xcoords.iloc[i + 1]
            xdifference = (x2 - x1)
            xdiff.append(xdifference)
        else:
            xdiff.append(np.nan)

    ydiff = []
    for i,value in enumerate (ycoords):
        if i < len(ycoords)-1:
            y1 = ycoords.iloc[i]
            y2 = ycoords.iloc[i + 1]
            ydifference = (y2 - y1)
            ydiff.append(ydifference)
        else:
            ydiff.append(np.nan)
    
    return (xdiff, ydiff)

def midpointx (pos_1, pos_2, pos_3, pos_4):
    '''
    give definition
    '''
    midpointx = (pos_1 + pos_3)/2
    midpointy = (pos_2 + pos_4)/2
    
    return (midpointx, midpointy)

#%%

# class for filtering automated data


def auto_scoring_tracefilter_full_CE(data,p=0.5,p_tail=15,p_head=5):
    #remove the not close to origin check(i don't know the meaning of it currently)
    mydata = data.copy()
    spine_column=['A_head',"F_spine1",'mid_spine1_spine2',"G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6","L_spine7","B_rightoperculum",
                 'E_leftoperculum']
    for i,c in enumerate(spine_column):
        #likelihood check
        likelihood_check=np.array(data[c]['likelihood']<p)
        mydata.loc[likelihood_check,(c,'x')]=np.nan;  mydata.loc[likelihood_check,(c,'y')]=np.nan
       
        #position difference check
        for j in ['x','y']:
            xdifference = abs(mydata[c][j].diff())
            if i<=3:
                xdiff_check = xdifference > p_head 
            else:
                xdiff_check = xdifference > p_tail
            mydata[c][j][xdiff_check] = np.nan
            
        #head and gill distance check
        if spine_column[i] in ['A_head',"B_rightoperculum",'E_leftoperculum']:
            if c=="A_head":
                #cal dist between head->spine1/operculum->head, if it's too large discard it
                head_spine1=np.array([mydata['A_head']['x']-mydata['F_spine1']['x'],mydata['A_head']['y']-mydata['F_spine1']['y']]).T
                head_dist=np.sqrt(np.sum(head_spine1*head_spine1,axis=1))
                head_dist_check=head_dist>25 #assume normal, use 3 \sigma rule
                mydata.loc[head_dist_check,(c,'x')] = np.nan; mydata.loc[head_dist_check,(c,'y')] = np.nan
         
            else:
                gill_head=np.array([mydata[c]['x']-mydata['A_head']['x'],mydata[c]['y']-mydata['A_head']['y']]).T
                gill_dist=np.sqrt(np.sum(gill_head*gill_head,axis=1))
                gill_dist_check=gill_dist>75
                mydata.loc[gill_dist_check,(c,'x')] = np.nan; mydata.loc[gill_dist_check,(c,'y')] = np.nan# it's just a random value, since it looks like the misclassified operculum
                #points are all deviant alot from the head
                
        #angle check
         #use the line from spine1 to mid point of spine1,spine2 as baseline, any segment to fit the spline should not have a angle
            #with the base line larger than 75 degree.
        if spine_column[i]=='mid_spine1_spine2':
            baseline=np.array([mydata['mid_spine1_spine2']['x']-mydata['F_spine1']['x'],mydata['mid_spine1_spine2']['y']-mydata['F_spine1']['y']]).T
        if spine_column[i] in ["G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6","L_spine7"]:
            orientation1=np.array([mydata[spine_column[i]]['x']-mydata[spine_column[i-1]]['x'],mydata[spine_column[i]]['y']-mydata[spine_column[i-1]]['y']]).T
            orientation2=np.array([mydata[spine_column[i]]['x']-mydata[spine_column[i-2]]['x'],mydata[spine_column[i]]['y']-mydata[spine_column[i-2]]['y']]).T
            orientation3=np.array([mydata[spine_column[i]]['x']-mydata[spine_column[i-3]]['x'],mydata[spine_column[i]]['y']-mydata[spine_column[i-3]]['y']]).T
            orientation=np.copy(orientation1); mask=np.isnan(np.sum(orientation,axis=1))           #if the previous point is already na, check the vector to the second closest point
            orientation[mask]=orientation2[mask]; mask=np.isnan(np.sum(orientation,axis=1))     # if it's also na,use next previous point ,and if the second closest point is also nan, stop here                        
            orientation[mask]=orientation3[mask]; mask=np.isnan(orientation) #if the previous 3 points are all NA, just.....don't use it for safety reason.
            safety_check=np.sum(mask,axis=1)!=0
            #if the baseline contains nan, skip this step
            inner_product =np.sum(baseline*orientation,axis=1)
            cos=inner_product/np.sqrt(np.sum(baseline*baseline,axis=1))/np.sqrt(np.sum(orientation*orientation,axis=1))
            angle=np.arccos(cos)/np.pi*180;angle_check=np.logical_or(np.logical_and(np.invert(np.isnan(angle)),angle>75),safety_check)          
            mydata.loc[angle_check,(c,'x')]=np.nan; mydata.loc[angle_check,(c,'y')]=np.nan           
    #check the orientation of head-spine1 in the end, I want to skip this step first, but some spline looks weird
    #so i implemented it, now the plot is better but the code looks weird
    orientation=-head_spine1
    inner_product =np.sum(baseline*orientation,axis=1)
    cos=inner_product/np.sqrt(np.sum(baseline*baseline,axis=1))/np.sqrt(np.sum(orientation*orientation,axis=1))
    angle=np.arccos(cos)/np.pi*180;angle_check=np.logical_and(np.invert(np.isnan(angle)),angle>75)
    mydata.loc[angle_check,('A_head','x')]=np.nan; mydata.loc[angle_check,('A_head','y')]=np.nan

    return mydata

def transform_data(df):
    mid_spine1_spine2=midpoint_wLikelihood(df['F_spine1']['x'],df['F_spine1']['y'],df['F_spine1']['likelihood'],
                                           df['G_spine2']['x'],df['G_spine2']['y'],df['G_spine2']['likelihood'])
    name_arr=[["mid_spine1_spine2","mid_spine1_spine2","mid_spine1_spine2"], ['x', 'y', 'likelihood']]
    mid_spine1_spine2=pd.DataFrame(mid_spine1_spine2,columns=pd.MultiIndex.from_arrays(name_arr,names=["bodyparts",'coords']))
    df=pd.concat([df,mid_spine1_spine2],axis=1)
    mid_spine2_spine3=midpoint_wLikelihood(df['G_spine2']['x'],df['G_spine2']['y'],df['G_spine2']['likelihood'],
                                           df['H_spine3']['x'],df['H_spine3']['y'],df['H_spine3']['likelihood'])
    name_arr=[["mid_spine2_spine3","mid_spine2_spine3","mid_spine2_spine3"], ['x', 'y', 'likelihood']]
    mid_spine2_spine3=pd.DataFrame(mid_spine2_spine3,columns=pd.MultiIndex.from_arrays(name_arr,names=["bodyparts",'coords']))
    df=pd.concat([df,mid_spine2_spine3],axis=1)
    return df

def midpoint_wLikelihood (x1, y1, l1,x2, y2,l2):
    '''
    give definition x1: x-value object 1, y1: y-value object 1, x2: x-value object 2
    y2: y-value object 2, l1: likelihood object 1,l2:likelihood object 2
    '''
    midpointx = (x1 + x2)/2
    midpointy = (y1 + y2)/2
    MinLikelihood=np.minimum(l1,l2) #the likelihood is set to the minimum of 2 columns

    return list(zip(midpointx, midpointy,MinLikelihood))


def filter_tailbeating(data,p0=50,p_head = 30, p1=25, p2 = 10, t1 = 20):
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
            dist_check = dist > p_head
            mydata[c]["x"][dist_check]  = np.nan
            mydata[c]["y"][dist_check]  = np.nan
        if (i>1 and i<(len(spine_column)-1)):
            r_decision = False
            dist1=np.sqrt(np.square(data[spine_column[i-1]]['x']-data[c]['x'])+np.square(data[spine_column[i-1]]['y']-data[c]['y']))
            dist2=np.sqrt(np.square(data[spine_column[i+1]]['x']-data[c]['x'])+np.square(data[spine_column[i+1]]['y']-data[c]['y']))
            # further check the relative position:
            if i > 2:
                dist3 = np.sqrt(np.square(data["F_spine1"]['x']-data[c]['x'])+np.square(data["F_spine1"]['y']-data[c]['y']))
                if np.logical_or((dist3[0] > ((i-1)*p1+t1)),(dist3[0]<((i-3)*p2))):
                    r_decision = True
            dist_check= np.logical_or(((dist1>p1)|(dist2>p1)), r_decision)
            mydata[c]["x"][dist_check] = np.nan
            mydata[c]["y"][dist_check] = np.nan
        if i==(len(spine_column)-1):
            dist1=np.sqrt(np.square(data[spine_column[i-1]]['x']-data[c]['x'])+np.square(data[spine_column[i-1]]['y']-data[c]['y']))
            dist2=np.sqrt(np.square(data[spine_column[i-2]]['x']-data[c]['x'])+np.square(data[spine_column[i-2]]['y']-data[c]['y']))
            dist3 = np.sqrt(np.square(data["F_spine1"]['x']-data[c]['x'])+np.square(data["F_spine1"]['y']-data[c]['y']))
            r_decision = np.logical_or(dist3[0]>((i-1)*p1), dist3[0]<((i-4)*p2))
            dist_check=np.logical_or(((dist1>p1)|(dist2>p1)), r_decision)
            mydata[c]["x"][dist_check] = np.nan
            mydata[c]["y"][dist_check] = np.nan
        
    return mydata

class Feature_extraction():
    
    '''
    #extract feature from given period
    #function to compare the cluster outputs of different features, the visualization is ultilized on the scatter plot of orientation and operculum angle
    #requires filtered df has the same schema as that predefined filtered_df, which should contain column
    #'A_head',"F_spine1",'mid_spine1_spine2',"G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6","L_spine7","B_rightoperculum",
    #'E_leftoperculum'
    '''
    
    def __init__(self,starttime=100000,endtime=None,duration=60):
        '''
        starttime: starttime of the period use
        duration: the duration of the period, the data will be sliced from starttime:starttime+40*duration
        '''
        self.starttime=starttime
        self.duration=duration
        if endtime==None:
            endtime=starttime+40*duration
        self.endtime=endtime
        
    def filter_df(self,raw_df,add_midpoint=True,p_head = 30,p_tail=15,p=0.5,p0=50,p1=25, p2 = 10, t1 = 20):
        '''
       
        Parameters
        ----------
        raw_df : TYPE 
            DESCRIPTION. Input raw DataFrame
        add_midpoint : TYPE, optional
            DESCRIPTION.  True:Yuyang's method, which will add 2 more columns(midpoint of spine1-spine2, and midpoint of spine2-spine3)
            False:Yuqi's method'

        Returns 
        -------
        filtered data
        '''

        if add_midpoint:
            df=transform_data(raw_df)
            return auto_scoring_tracefilter_full_CE(df,p_head=p_head,p_tail=p_tail,p=p)
        else:
            return filter_tailbeating(raw_df,p0=p0,p_head = p_head, p1=p1, p2 = p2, t1 = t1)
            
        
    def fit(self,filtered_df,filter_feature=True,fill_na=True,estimate_na=True):
        '''
       this function computes all the features we have thought about
        Parameters
        ----------
        filtered_df : dataframe
            filtered dataframe
        filter_feature: filter out extreme points in the feature calculated, basically it removes points which violates the 
        3 /sigma rule
        fill_na: whether to fill na after the features are calculated
        estimate_na: whether to estimate the nas before fitting the spline, assuming the missing point is on the line of it's
        previous and next available point

        Returns 
        -------
      
        '''
        starttime=self.starttime
        endtime=self.endtime
        trunc_df=filtered_df.loc[starttime:endtime-1,:]
        operculum=auto_scoring_get_opdeg(trunc_df)
        
        ori=orientation(trunc_df)
    
        turn_angle=turning_angle_spine(trunc_df)
        
        mov_speed=speed(trunc_df)
        
        if "mid_spine1_spine2" in filtered_df.columns:
            y=trunc_df.loc[:,[('A_head','y'),("F_spine1","y"),('mid_spine1_spine2',"y"),("G_spine2","y"),
                            ('mid_spine2_spine3',"y"),("H_spine3","y"),("I_spine4","y"),("J_spine5","y"),
                            ("K_spine6","y"),("L_spine7","y")]]
            x=trunc_df.loc[:,[('A_head','x'),("F_spine1","x"),('mid_spine1_spine2',"x"),("G_spine2","x"),
                            ('mid_spine2_spine3',"x"),("H_spine3","x"),("I_spine4","x"),("J_spine5","x"),
                            ("K_spine6","x"),("L_spine7","x")]]
        else:
            y=trunc_df.loc[:,[('A_head','y'),("F_spine1","y"),("G_spine2","y"),
                            ("H_spine3","y"),("I_spine4","y"),("J_spine5","y"),
                            ("K_spine6","y"),("L_spine7","y")]]
            x=trunc_df.loc[:,[('A_head','x'),("F_spine1","x"),("G_spine2","x"),
                            ("H_spine3","x"),("I_spine4","x"),("J_spine5","x"),
                            ("K_spine6","x"),("L_spine7","x")]]
        #By first stacking the data we want to a 3D array, the running time decreases significantly!
        #using for gives almost same run time
        concat_array=np.stack((x,y),axis=-1)
        if estimate_na:
            #give estimate to na's in 
            concat_array=np.array(list(map(self.linearly_fill_data,concat_array)))
        map_results=np.array(list(map(self.cal_curvature_and_dir,concat_array)))
        curvatures=pd.DataFrame(map_results[:,0,:])
        cos=pd.DataFrame(map_results[:,1,:])
        diff_curvature=pd.DataFrame(np.diff(curvatures,axis=0,prepend=np.expand_dims(curvatures.loc[0,:],0)))
        if "mid_spine1_spine2" in filtered_df.columns:
            curvatures.columns=["curvature_head","curvature_spine1","curvature_spine1.5","curvature_spine2","curvature_spine2.5","curvature_spine3",
              "curvature_spine4","curvature_spine5","curvature_spine6","curvature_spine7"]
            diff_curvature.columns=["diff_curvature_head","diff_curvature_spine1","diff_curvature_spine1.5","diff_curvature_spine2","diff_curvature_spine2.5","diff_curvature_spine3",
              "diff_curvature_spine4","diff_curvature_spine5","diff_curvature_spine6","diff_curvature_spine7"]
            cos.columns=["tangent_head","tangent_spine1","tangent_spine1.5","tangent_spine2","tangent_spine2.5","tangent_spine3",
              "tangent_spine4","tangent_spine5","tangent_spine6","tangent_spine7"]
        else:
            curvatures.columns=["curvature_head","curvature_spine1","curvature_spine2","curvature_spine3",
              "curvature_spine4","curvature_spine5","curvature_spine6","curvature_spine7"]
            diff_curvature.columns=["diff_curvature_head","diff_curvature_spine1","diff_curvature_spine2","diff_curvature_spine3",
              "diff_curvature_spine4","diff_curvature_spine5","diff_curvature_spine6","diff_curvature_spine7"]
            cos.columns=["tangent_head","tangent_spine1","tangent_spine2","tangent_spine3",
              "tangent_spine4","tangent_spine5","tangent_spine6","tangent_spine7"]
        if filter_feature:
            #filter feature according to "3 sigma" rule, that is, I assume the features follows normal distribution, and 
            # I filter out points where it's distance to its mean is greater than 3xstd
            operculum[abs(operculum-np.nanmean(operculum,axis=0))>3*np.nanstd(operculum,axis=0)]=np.nan
            ori[abs(ori-np.nanmean(ori,axis=0))>3*np.nanstd(ori,axis=0)]=np.nan
            turn_angle[abs(turn_angle-np.nanmean(turn_angle,axis=0))>3*np.nanstd(turn_angle,axis=0)]=np.nan
            mov_speed[abs(mov_speed-np.nanmean(mov_speed,axis=0))>3*np.nanstd(mov_speed,axis=0)]=np.nan
            curvatures[abs(curvatures-np.nanmean(curvatures,axis=0))>3*np.nanstd(curvatures,axis=0)]=np.nan
            diff_curvature[abs(diff_curvature-np.nanmean(diff_curvature,axis=0))>3*np.nanstd(diff_curvature,axis=0)]=np.nan
            cos[abs(cos-np.nanmean(cos,axis=0))>3*np.nanstd(cos,axis=0)]=np.nan
            
        #then deal with the NAs in the feature
        if fill_na==True:       
            curvatures=curvatures.fillna(method='ffill'); curvatures=curvatures.fillna(curvatures.mean())
            cos=cos.fillna(method='ffill').fillna(cos.mean())
            diff_curvature=diff_curvature.fillna(method='ffill'); diff_curvature=diff_curvature.fillna(diff_curvature.mean())
            operculum=operculum.fillna(method="ffill").fillna(operculum.mean())
            ori=pd.Series(ori).fillna(method='ffill').fillna(np.nanmean(ori))
            mov_speed=pd.Series(mov_speed).fillna(method="ffill").fillna(np.nanmean(mov_speed))
            turn_angle=pd.Series(turn_angle).fillna(method="ffill").fillna(np.nanmean(turn_angle))
        self.curvatures=curvatures
        self.cos=cos
        self.curvatures=curvatures
        self.diff_curvature=diff_curvature
        self.operculum=operculum
        self.ori=ori
        self.mov_speed=mov_speed
        self.turn_angle=turn_angle
    
    def linearly_fill_data(self,x):
        ##Yuqi's code, filter step is skipped, I will just show na in the curvatures computed
        not_na = np.unique(np.where(~np.isnan(x))[0])
        if (len(not_na)>=4):
            h = not_na[0]
            s1 = not_na[1]
            if (h == 0) & (s1==1):
                for j in range(len(not_na)):
                    if j > 1:
                        current = not_na[j]
                        pre = not_na[j-1]
                        point = current-pre
                        if point > 1:
                            #if there's point missing in consective samples for spline,fill the missing points in 
                            #the middle with a linear estimate
                            dx = x[current][0]-x[pre][0]
                            dy = x[current][1]-x[pre][1]
                            for k in range(1, point):
                                x[pre+k][0] = x[pre][0]+k*dx/point
                                x[pre+k][1] = x[pre][1]+k*dy/point
        return x
    def cal_curvature_and_dir(self,x):
        pts=x
        line=pts[2]-pts[1]
        index=~np.isnan(pts).any(axis=1)
        pts=pts[index]
        curvature=np.repeat(np.nan,x.shape[0])
        directions=np.repeat(np.nan,x.shape[0])
        if(pts.shape[0]>=4):
            tck,u=interpolate.splprep(pts.T, u=None, s=0.0)
            dx1,dy1=interpolate.splev(u,tck,der=1)
            dx2,dy2=interpolate.splev(u,tck,der=2)
            k=(dx1*dy2-dy1*dx2)/np.power((np.square(dx1)+np.square(dy1)),3/2)
            direction=(dy1*line[1]+dx1*line[0])/np.linalg.norm(line)/np.sqrt(dy1*dy1+dx1*dx1)
            directions[index]=direction
            curvature[index]=k
        return [curvature,directions]
    
    def export_df(self):
        #combine curvature/diff_curvature/tangent_angle and other features to one df
        other_features=pd.DataFrame({"operculum":np.array(self.operculum),"orientation":self.ori,"movement_speed":np.array(self.mov_speed),
                                     "turning_angle":self.turn_angle},index=self.curvatures.index)
        curvatures=self.curvatures
        diff_curvatures=self.diff_curvature
        tangent=self.cos
        return other_features,curvatures,diff_curvatures,tangent
        
    def visualize_cluster(self,num_cluster=2,dpi=300,s=2,cmap='cividis'):
        '''
        dpi: the resolution of image?
        num_cluster: the number of cluster in kmeans
        s:size of pts
        cmap:cmap attribute in plt
        NOT REVISED YET, DONT USE THAT
        '''
        #scale them
        from sklearn.preprocessing import StandardScaler
        operculum=self.operculum;ori=self.ori
        filled_curvatures=self.curvatures;diff_curvature=self.diff_curvature;filled_cos=self.cos
        scaler = StandardScaler();scaler.fit(filled_curvatures);filled_curvatures=pd.DataFrame(scaler.transform(filled_curvatures))   
        scaler = StandardScaler();scaler.fit(diff_curvature); diff_curvature=pd.DataFrame(scaler.transform(diff_curvature))
        scaler = StandardScaler();scaler.fit(filled_cos);filled_cos=pd.DataFrame(scaler.transform(filled_cos)) 
        #kmeans cluster, probably not the optimal way to do this
        from sklearn.cluster import KMeans
        kmeans_curvature = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=1000, n_init=10);kmeans_curvature.fit(filled_curvatures)
        kmeans_diffCurvature = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=1000, n_init=10); kmeans_diffCurvature.fit(diff_curvature)
        kmeans_cos = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=1000, n_init=10);kmeans_cos.fit(filled_cos)
    
        #visualize
        #it's not the most solid way to do this, just want to check the features don't look uniform distributed in the plot
        label_curvature=kmeans_curvature.predict(filled_curvatures)
        label_diffCurvature=kmeans_diffCurvature.predict(diff_curvature)
        label_cos=kmeans_cos.predict(filled_cos)
        self.label_curvature=label_curvature
        self.label_diffCurvature=label_diffCurvature
        self.label_cos=label_cos
        fig=plt.figure(dpi=dpi)
        ax=fig.add_subplot(1,3,1)
        ax.title.set_text("curvature")
        ax.scatter(x=operculum,y=ori,s=s, c=label_curvature, cmap='cividis')
        ax=fig.add_subplot(1,3,2)
        ax.scatter(x=operculum,y=ori,s=s, c=label_diffCurvature, cmap='cividis')
        ax.title.set_text("diff_curvature")
        ax=fig.add_subplot(1,3,3)
        ax.scatter(x=operculum,y=ori,s=s, c=label_cos, cmap='cividis')
        ax.title.set_text("tangent line")
        print("cluster on {} groups".format(num_cluster))

##The old name
class features(Feature_extraction):
    def  __init__(self,starttime=100000,endtime=None,duration=600):
        super(features,self).__init__(starttime,endtime,duration)
        
        
#%%

# functions for scorign automated data
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


def custom_thresh(Looped):
    min_list = []
    
    for n in np.arange(len(Looped.columns)):
        plot = sns.kdeplot(Looped[Looped.columns[n]], legend = True)
    
    for i in np.arange(len(plot.lines)):
        x = plot.lines[i].get_xdata()
        y = plot.lines[i].get_ydata()
        
        zip_xy = pd.DataFrame(np.array(list(zip(x,y))))
        
        
        min_x = zip_xy[zip_xy.columns[0]].iloc[argrelextrema(zip_xy[zip_xy.columns[1]].values, np.less)]
        max_x = zip_xy[zip_xy.columns[0]].iloc[argrelextrema(zip_xy[zip_xy.columns[1]].values, np.greater)]
        
        min_y = zip_xy[zip_xy.columns[1]].iloc[argrelextrema(zip_xy[zip_xy.columns[1]].values, np.less)]
        max_y = zip_xy[zip_xy.columns[1]].iloc[argrelextrema(zip_xy[zip_xy.columns[1]].values, np.greater)]
        
        zip_min_xy = pd.DataFrame(np.array(list(zip(min_x,min_y))))
        zip_max_xy = pd.DataFrame(np.array(list(zip(max_x,max_y))))
        
       
        if len(zip_max_xy) == 2:
            for j in np.arange(len(zip_min_xy)):
                if zip_min_xy[0][j] < zip_max_xy[0][len(zip_max_xy)-1]:
                    if zip_min_xy[0][j] > zip_max_xy[0].iloc[np.argmax(zip_max_xy[1])]:
                        plt.axvline(zip_min_xy[0][j], color = 'red')
                        min_list.append(zip_min_xy[0][j])
                        print(zip_min_xy[0][j], i)
    
    # custom_thresh = np.array(min_list).sum()/len(min_list)
    custom_thresh = np.average(min_list)
    custom_thresh_median = np.median(min_list)
    plt.axvline(custom_thresh, color = 'blue')
    return(custom_thresh)
# SCRAP coding for if you did the whole testing period at once
# x_all = sns.kdeplot(Oper_Angle_Test).lines[0].get_xdata()
# y_all = sns.kdeplot(Oper_Angle_Test).lines[0].get_ydata()
# zip_xy_all = pd.DataFrame(np.array(list(zip(x_all,y_all))))
# min_x_all = zip_xy_all[zip_xy_all.columns[0]].iloc[argrelextrema(zip_xy_all[zip_xy_all.columns[1]].values, np.less_equal)]
# for j in np.arange(len(min_x_all)):
#     plt.axvline(min_x_all.values[j], color = 'red')
#%%

# plotting functions for 1) means 2) index 3) binned data 4) event plots 5) density plots

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def uniform(indbin,LoopLength):
    A = len(indbin)
    B = LoopLength
    C = B - A
    end = np.zeros(C)
    new = np.concatenate((indbin, end))
    
    return(new)

def binarizeOp(Operangle, threshold):
    

    boolean = Operangle.apply(lambda x: 1 if x > threshold else 0).values
#    boolean = binary_dilation(boolean, structure = np.ones(40,))
    
    binindex = np.where(boolean)[0]

    return (binindex)

#%%

# Making videos
# ADDITIONAL VIDEO MAKING SCRIPTS CAN BE FOUND AT GitHub/DSI_Students/SegregatingBehavior.py 

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

#%%

#HMM associated functions for performing PCA and HMM


#%%

# functions for manual scoring

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




def binarize_Op_2(Operangle, lb = 65, ub = 135):
    boolean = Operangle.apply(lambda x: 1 if lb < x < ub else 0)
    return(boolean)


#%%

#ROC Functions, can find more at DSI_Usefulfunctions.py

def auto_scoring_TS1(data_auto,thresh_param0 = 70,thresh_param1 = 180):
    '''
    Function to automatically score operculum as open or closed based on threshold parameters. 
    
    Parameters: 
    data_auto: traces of behavior collected as a pandas array. 
    thresh_param0: lower threshold for operculum angle
    thresh_param1: upper threshold for operculum angle
    
    Returns:
    pandas array: binary array of open/closed scoring
    '''
    degree = auto_scoring_get_opdeg(data_auto)
    return degree.apply(lambda x: 1 if thresh_param0 < x < thresh_param1 else 0)
##############################################################################

##############################
def auto_scoring_M2(data_auto,thresh_param0 = 70,thresh_param1 = 180,thresh_param3 = 30):
    
    raw_out = auto_scoring_TS1(data_auto,thresh_param0,thresh_param1)
    
    widthfilter_out = auto_scoring_widthfilter(raw_out,widththresh = 30)
    return widthfilter_out


## 4/27/19: Updated parameter list handling to accomodate arbitrary lists of lists, and reshape them in a way that can 
## be iterated through intuitively. 
def manip_paramlist(Plist):
    '''
    A function that takes in an arbitrary parameter list (list of lists) and returns a 2d array that can be iterated 
    on directly. 
    
    Parameters: 
    Plist [list]: a list of lists containing valid values for each parameter we consider. 
    
    Returns:
    array: a 2d array containing all possible parameter combinations as rows.
    array: a 2d array containing the relevant indices of all possible parameter combinations. 
    '''
    ## We will also return an array for indexing purposes:
    Pindex = [np.arange(len(plist)) for plist in Plist]
    
    meshgrid,meshgrid_index = np.meshgrid(*Plist,indexing = 'ij'),np.meshgrid(*Pindex,indexing = 'ij') ## Returns a list of Nd arrays, where N is the number of lists we consider. 

    Parray,Parray_index = np.array(meshgrid).T.reshape(-1,len(Plist)),np.array(meshgrid_index).T.reshape(-1,len(Plist))
    return Parray,Parray_index

## Takes together a function from automatic data -> binary scores, and a list of relevant parameters, as well as 
## manual data and aforementioned automatically scored data

def NC_ROC(data_auto, params):
    p0 = params[0]
    p1 = params[1]
    return auto_scoring_TS1(data_auto,p0,p1)

def TS1_ROC(data_auto,params):
    p0 = params[0]
    p1 = params[1]
    return auto_scoring_TS1(data_auto,p0,p1)

def Yuyang_ROC(Yuyang_filter,params):
    lb = params[0]
    ub = params[1]
    return binarize_Op_2(Yuyang_filter, lb, ub)

def M2_ROC(data_auto,params):
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    return auto_scoring_widthfilter(auto_scoring_TS1(data_auto, p0, p1), p2)

def F_Auto(data_auto,params):
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    return auto_scoring_TS1(auto_scoring_tracefilter(data_auto, p2, p3, p4, p5, p6), p0, p1)

def F_Auto_M2(data_auto,params):
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    return auto_scoring_widthfilter(auto_scoring_TS1(auto_scoring_tracefilter(data_auto, p3, p4, p5, p6, p7), p0, p1), p2)


# ##### Defining Generalizable ROC Generating Functions



def ROC_Analysis_vec(func,Plist,data_manual,data_auto):
    
    ## Reshape Plist to accept all combinations: 
    Parray,Parray_index = manip_paramlist(Plist)
    
    Plist_shape = [len(plist) for plist in Plist]
    FPR = np.zeros(Plist_shape)
    TPR = np.zeros(Plist_shape)
    FDR = np.zeros(Plist_shape)
    Pr = np.zeros(Plist_shape)
    YoudenR = np.zeros(Plist_shape)
    
    ## Process manual data once: 
    data_manual_open = manual_scoring(data_manual,data_auto)
    
    for ip,params in tqdm(enumerate(Parray)):
        data_auto_open = func(data_auto,params)

        TP_vec = data_manual_open.values*data_auto_open.values[:-1]

        TP = sum(abs(TP_vec))

        TotalPos = data_auto_open.sum()
        FP = TotalPos - TP
        FP

        TN_vec = (1-data_manual_open.values)*(1-data_auto_open.values)[:-1]

        TN = sum(abs(TN_vec))


        TotalNeg = len(data_auto_open) - TotalPos
        FN = TotalNeg - TN
        FN

        TPR1 = TP / (TP + FN)
        FPR1 = FP/(FP + TN)
        Precision1 = TP/(TP + FP)
        FalseDiscovery1 = FP/(FP + TP) 
        Spec = TN/(TN + FP)
        Youden = TPR1+Spec-1

        FPR[tuple(Parray_index[ip,:])] = FPR1
        TPR[tuple(Parray_index[ip,:])] = TPR1
        FDR[tuple(Parray_index[ip,:])] = FalseDiscovery1
        Pr[tuple(Parray_index[ip,:])] = Precision1
        YoudenR[tuple(Parray_index[ip,:])] = Youden
        print(Pr)
        

    return TPR,FPR,Pr,FDR,YoudenR







 