#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 21:26:50 2020

@author: miaoyuqi
"""
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Conspecific_Tracking_ta_edit_117 as cs
from datetime import datetime
from scipy import interpolate 
from scipy import misc
from beating import rotation
from beating import tail_spline
from functions import *
from hmmlearn import hmm
from auto_filter_full import  auto_scoring_tracefilter_full, transform_data
from functools import partial
from find_features import Feature_extraction
import seaborn as sn
import ssm
from sklearn import metrics

## visulization and clustering methods

def preprocessing(df):
    x = df.values
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)
    return(x,df.columns)

plt.rcParams["axes.grid"] = False

def PCA_period(x,names,period = range(90000,150000)):
    x = x[period, :]
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(x)
    pcs=pca.transform(x)
    
    # variance explained PC
    fig = plt.figure(figsize=(4, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    for a,b in zip(range(x.shape[1]),np.cumsum(pca.explained_variance_ratio_)):
        plt.text(a, b, round(b,2), ha='center', va='top', fontsize=15)
    #plt.axvline(x = 4,color = "r")
    plt.show()
    
     # first 2 pc
    plt.scatter(pcs[:,0],pcs[:,1],s = 1)
    plt.show()
    
    # loading plot
    loadings = pd.DataFrame(pca.components_, columns = names)
    loadings_abs = abs(loadings)
    import seaborn as sn
    sn.heatmap(loadings_abs, annot=True,cmap = "YlGnBu")
    plt.show()
    
    return(pcs)
'''
cluster method -- kmeans and hmm:
1. using scaled points to fit the clustering;
2. using pcs to visulize the clustering resul
'''

def kmeans_period_plot(x,pcs,n_cluster = 3,dim = 3,period = range(90000,150000),elev=45,azim=45,score = False):
    x = x[period,:]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=300, n_init=10)
    kmeans.fit(x)
    y=kmeans.predict(x)
    if dim == 2:
        plt.scatter(pcs[:,0],pcs[:,1], c = y, s = 1)
    if dim == 3:
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(pcs[:, 0], pcs[:, 1],pcs[:, 2], s=1,c = y,cmap='viridis')
        ax1.set_xlim(-3,3)
        ax1.set_ylim(-3,3)
        ax1.set_zlim(-3,3)
        ax1.view_init(elev=elev,azim=azim)
#    # cluster matric
#    if score == True:
#        s = metrics.silhouette_score(x, y, metric='euclidean')
#        print("the silhouette_score is " + str(s))
    return(y)

def hmm_period_plot(x,pcs,n_cluster = 3,visual_dim = 3,n_iters = 20,period = range(90000,150000),elev=45,azim=45,score = False):
    x = x[period,:]
    D = x.shape[1]
#     hmm_3=hmm.GaussianHMM(n_components=n_cluster, covariance_type="full", n_iter=100)
#     hmm_3.fit(x)
    hmm = ssm.HMM(n_cluster, D, observations="gaussian")
    
    # learned matrix
    fig = plt.figure(figsize=(4, 4))
    learned_transition_mat = hmm.transitions.transition_matrix
    im = plt.imshow(learned_transition_mat, cmap='viridis')
    plt.title("Learned Transition Matrix")

    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    
    # get iteration likelihood
    N_iters = n_iters
    hmm_lls = hmm.fit(np.array(x), method="em", num_em_iters=N_iters)
    print("likelihood of em iteration is" + str(hmm_lls[len(hmm_lls)-1]))
    
    # get states distribution
    hmm_state = hmm.most_likely_states(x)
    plt.imshow(hmm_state[None,:], aspect="auto", cmap="YlGnBu")
    plt.xlim(0,len(period))
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.xlabel("time")
    plt.show()
    
    # get state duration distribution
    fig = plt.figure(figsize=(4, 4))
    inferred_state_list, inferred_durations = ssm.util.rle(hmm_state)
    inf_durs_stacked = []
    for s in range(n_cluster):
        inf_durs_stacked.append(inferred_durations[inferred_state_list == s])
    plt.hist(inf_durs_stacked, label=['state ' + str(s) for s in range(n_cluster)])
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.xlim(0,120)
    plt.legend()
    plt.title('Histogram of Inferred State Durations')
    plt.show()
    
    # get pca clustering visualization
    if visual_dim == 2:
        plt.scatter(pcs[:,0],pcs[:,1], c = hmm_state, s = 1)
        plt.show()
    if visual_dim == 3:
        
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(pcs[:, 0], pcs[:, 1],pcs[:, 2], s=1,c = hmm_state,cmap='viridis')
        ax1.set_xlim(-3,3)
        ax1.set_ylim(-3,3)
        ax1.set_zlim(-3,3)
        ax1.view_init(elev=elev,azim=azim)
#    # cluster metric
#    if score == True:
#        s = metrics.silhouette_score(x, y, metric='euclidean')
#    print("the silhouette_score is " + str(s))
    
    return(hmm_state)

def cluster_PC_density(pcs,cluster,pc = "PC1"):
    feature_df = pd.DataFrame(pcs)
    feature_df.columns = ["{}{}".format("PC",i) for i in range(1,feature_df.shape[1]+1)]
    feature_df["cluster"] = cluster
    feature_df.groupby(cluster)
    feature_df.groupby(cluster)[pc].plot(kind = "kde",legend = True,title = pc)
    