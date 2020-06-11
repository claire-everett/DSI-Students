#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:36:18 2020

@author: ryan
"""

from functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate 
class features():
    '''
    #function to compare the cluster outputs of different features, the visualization is ultilized on the scatter plot of orientation and operculum angle
    #requires filtered df has the same schema as that predefined filtered_df, which should contain column
    #'A_head',"F_spine1",'mid_spine1_spine2',"G_spine2",'mid_spine2_spine3',"H_spine3","I_spine4","J_spine5","K_spine6","L_spine7","B_rightoperculum",
    #'E_leftoperculum'
    '''
    
    def __init__(self,starttime=100000,duration=60):
        '''
        starttime: starttime of the period use
        duration: the duration of the period, the data will be sliced from starttime:starttime+40*duration
        '''
        self.starttime=starttime
        self.duration=duration
    def fit(self,filtered_df,fill_na=True):
        #compute operculum angle and orientation
        starttime=self.starttime
        duration=self.duration
        operculum=auto_scoring_get_opdeg(filtered_df)
        operculum=operculum[starttime:starttime+duration*40]
        operculum[abs(operculum-np.nanmean(operculum,axis=0))>3*np.nanstd(operculum,axis=0)]=np.nan
        ori=orientation(filtered_df)
        ori=ori[starttime:starttime+duration*40]
        ori[abs(ori-np.nanmean(ori,axis=0))>3*np.nanstd(ori,axis=0)]=np.nan
        turn_angle=turning_angle_spine(filtered_df)
        turn_angle=turn_angle[starttime:starttime+duration*40]
        turn_angle[abs(turn_angle-np.nanmean(turn_angle,axis=0))>3*np.nanstd(turn_angle,axis=0)]=np.nan
        mov_speed=speed(filtered_df)
        mov_speed=mov_speed[starttime:starttime+duration*40]
        mov_speed[abs(mov_speed-np.nanmean(mov_speed,axis=0))>3*np.nanstd(mov_speed,axis=0)]=np.nan
        curvatures=[]
        baseline=np.array([filtered_df['mid_spine1_spine2']['x']-filtered_df['F_spine1']['x'],filtered_df['mid_spine1_spine2']['y']-filtered_df['F_spine1']['y']]).T
        cos_derivative_baseline=[]
    #calculate the 3 features(curvature/diff_curvature/cos between the tangent line and baseline(spine1->spine1.5))
        for i in range(starttime,starttime+duration*40):
            line=baseline[i,:]
            y=filtered_df.loc[i,[('A_head','y'),("F_spine1","y"),('mid_spine1_spine2',"y"),("G_spine2","y"),
                            ('mid_spine2_spine3',"y"),("H_spine3","y"),("I_spine4","y"),("J_spine5","y"),
                            ("K_spine6","y"),("L_spine7","y")]]
            x=filtered_df.loc[i,[('A_head','x'),("F_spine1","x"),('mid_spine1_spine2',"x"),("G_spine2","x"),
                            ('mid_spine2_spine3',"x"),("H_spine3","x"),("I_spine4","x"),("J_spine5","x"),
                            ("K_spine6","x"),("L_spine7","x")]]
            pts=np.vstack([x,y]).T
            index=~np.isnan(pts).any(axis=1)
            pts=pts[index]
            curvature=np.repeat(np.nan,10)
            cos=np.repeat(np.nan,10)
            if(pts.shape[0]>=4):
                tck,u=interpolate.splprep(pts.T, u=None, s=0.0)
                dx1,dy1=interpolate.splev(u,tck,der=1)
                dx2,dy2=interpolate.splev(u,tck,der=2)
                k=(dx1*dy2-dy1*dx2)/np.power((np.square(dx1)+np.square(dy1)),3/2)
                cos_=(dy1*line[1]+dx1*line[0])/np.linalg.norm(line)/np.sqrt(dy1*dy1+dx1*dx1)
                cos[index]=cos_
                curvature[index]=k
                curvatures.append(curvature)
                cos_derivative_baseline.append(cos)
            else:
                curvatures.append(curvature)
                cos_derivative_baseline.append(cos)
        
        
            
        #filter out value that is likely to be outlier
        curvatures=np.array(curvatures); curvatures=np.vstack(curvatures);curvatures=pd.DataFrame(curvatures)
        curvatures[abs(curvatures-np.nanmean(curvatures,axis=0))>3*np.nanstd(curvatures,axis=0)]=np.nan
        diff_curvature=pd.DataFrame(np.diff(curvatures,axis=0,prepend=np.expand_dims(curvatures.loc[0,:],0)))
        diff_curvature[abs(diff_curvature-np.nanmean(diff_curvature,axis=0))>3*np.nanstd(diff_curvature,axis=0)]=np.nan
        cos_derivative_baseline=pd.DataFrame(np.vstack(np.array(cos_derivative_baseline)))
        cos_derivative_baseline[abs(cos_derivative_baseline-np.nanmean(cos_derivative_baseline,axis=0))>3*np.nanstd(cos_derivative_baseline,axis=0)]=np.nan
        #then deal with the NAs in the feature
        if fill_na==True:       
            filled_curvatures=curvatures.fillna(method='ffill'); filled_curvatures=curvatures.fillna(curvatures.mean())
            filled_cos=cos_derivative_baseline.fillna(method='ffill').fillna(cos_derivative_baseline.mean())
            diff_curvature=diff_curvature.fillna(method='ffill'); diff_curvature=diff_curvature.fillna(diff_curvature.mean())
            operculum=operculum.fillna(method="ffill").fillna(operculum.mean())
            ori=pd.Series(ori).fillna(method='ffill').fillna(np.nanmean(ori))
            mov_speed=pd.Series(mov_speed).fillna(method="ffill").fillna(np.nanmean(mov_speed))
            turn_angle=pd.Series(turn_angle).fillna(method="ffill").fillna(np.nanmean(turn_angle))
            self.curvatures=filled_curvatures
            self.cos=filled_cos
        else:
            self.curvatures=curvatures
            self.cos=cos_derivative_baseline
        self.diff_curvature=diff_curvature
        self.operculum=operculum
        self.ori=ori
        self.mov_speed=mov_speed
        self.turn_angle=turn_angle
    def visualize_cluster(self,num_cluster=2,dpi=300,s=2,cmap='cividis'):
        '''
        dpi: the resolution of image?
        num_cluster: the number of cluster in kmeans
        s:size of pts
        cmap:cmap attribute in plt
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