#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:13:49 2020

@author: ryan
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate 
from moviepy.video.io.bindings import mplfig_to_npimage
def make_frame_factory(starttime,duration,fps,raw_df,filtered_df):
    fig, ax = plt.subplots(dpi=300)
    focus_df=raw_df
    def make_frame_withgill(time):
        t=np.int(starttime+time*fps)
        ax.clear()
        test_point = filtered_df.iloc[t,:]
        raw_test_point=focus_df.iloc[t,:]
        x = test_point[[0, 3, 6, 9, 12, 15,18,21,24,27]]
        y = test_point[[1, 4, 7, 10, 13, 16,19,22,25,28]]
        pts=np.vstack([x,y]).T
        ox = raw_test_point[[0, 3, 6, 9, 12, 15,18,21,24,27]]
        oy = raw_test_point[[1, 4, 7, 10, 13, 16,19,22,25,28]]
        raw_pts=np.vstack([ox,oy]).T
        if ~(np.isnan(pts[0,0]) or np.isnan(pts[0,1])):
            ax.text(raw_pts[0,0],raw_pts[0,1],"h",fontsize=12)
        if ~(np.isnan(test_point[30]) or np.isnan(test_point[31]) or np.isnan(test_point[33]) or np.isnan(test_point[34])):
            ax.scatter(test_point[[30,33]],test_point[[31,34]],s=10,color="purple")
            ax.plot(test_point[[30,0]],test_point[[31,1]],'b-',linewidth=2)
            ax.plot(test_point[[33,0]],test_point[[34,1]],'b-',linewidth=2)
        pts=pts[~np.isnan(pts).any(axis=1)]
        colormap = np.array(['b', 'brown', 'green','red','grey','pink','orange','black','gold','yellow'])
        categories = np.array(range(raw_pts.shape[0]))
        ax.scatter(raw_pts[:,0], raw_pts[:,1],s=10,c=colormap[categories])
        if(pts.shape[0]>=4):
            tck,u=interpolate.splprep(pts.T, u=None, s=0.0) 
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = interpolate.splev(u_new, tck, der=0)
            ax.plot(x_new, y_new, 'b-',linewidth=2)
        ax.set_ylim([0,500])
        ax.set_xlim([0,500])
        ax.grid(b=None)
        ax.patch.set_facecolor('white')
    #ax.axis("off")
        return mplfig_to_npimage(fig)
    return make_frame_withgill