#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:21:18 2020

@author: Claire
"""

#%% 
## Import necessary Packages/Functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns; sns.set()
from DSI_UsefulFunctions import mydistance, lawofcosines, speed, myvelocity,  midpointx, manual_scoring,coords, auto_scoring_get_opdeg, auto_scoring_smooth_opdeg,auto_scoring_widthfilter, auto_scoring_tracefilter,auto_scoring_TS1, auto_scoring_M2, manip_paramlist, TS1_ROC, M2_ROC, F_Auto, F_Auto_M2, ROC_Analysis_vec   
#%%
## Load Auto and Manual Data
#Auto
h5_dir = '/Users/Claire/Desktop/FishBehavioralAnalysis'
h5_files = glob(os.path.join(h5_dir,'*.h5'))
print(h5_files)

file_handle = h5_files[1]

with pd.HDFStore(file_handle,'r') as help2:
    data_auto = help2.get('df_with_missing')
    data_auto.columns= data_auto.columns.droplevel()
#%%
#Manual
excel_dir = '/Users/Claire/Desktop/FishBehavioralAnalysis'
excel_files = glob(os.path.join(h5_dir,'*.xlsx'))
print(excel_files)

file_handle = excel_files[-2]
data_manual = pd.read_excel(file_handle)


#%%

## Perform Heatmap analysis
# first crop the data to feasible analysis size
data_auto_crop = data_auto.copy()[:1400]
Compare = pd.DataFrame(0, index=np.arange(len(data_auto_crop)), columns = ['Manual', 'Automatic','M2','F_Auto', 'F_Auto_M2'])
#%%
Compare['Manual'] = manual_scoring(data_manual,data_auto_crop, crop0 = 0, crop1 = 1400)
Compare['Automatic'] = auto_scoring_TS1(data_auto_crop, 65, 135)
Compare['M2'] =  auto_scoring_M2(data_auto_crop,65,135,10)
Compare['F_Auto'] = auto_scoring_TS1(auto_scoring_tracefilter(data_auto_crop), 65, 135)
Compare['F_Auto_M2'] = auto_scoring_widthfilter(auto_scoring_TS1(auto_scoring_tracefilter(data_auto), 65, 180), 10)

ax = sns.heatmap(Compare)
#%%
## Perform ROC analysis
Plist = [np.linspace(40,180,100),np.linspace(80,180,10)]
ROCoutputTS1 = ROC_Analysis_vec(TS1_ROC,Plist,data_manual,data_auto)

TPR = ROCoutputTS1[0]
FPR = ROCoutputTS1[1]
Youden = ROCoutputTS1[4]
for i in range(len(Plist[1])):
    t = TPR[:,i]
    f = FPR[:,i]
    plt.plot(f,t,label = 'ub = '+str(np.round(Plist[1][i],1)))
    
    plt.legend(loc = 0)
plt.plot(np.linspace(0,1,10),np.linspace(0,1,10),'--')
# plt.xlim([0,0.5])
# plt.ylim([0,1])
# plt.axis('equal')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Youden\'s J =' +str(np.round(np.max(Youden),2)))
plt.savefig('Raw ROC')
plt.show()

#%%
# Find lower bound and upper bound that provide the best signal for fish and see how much variation
z = np.unravel_index(np.argmax(Youden),shape = (len(Plist[0]),len(Plist[1])))

TPmax,FPmax = TPR[z],FPR[z]
print(TPmax, FPmax)

print('lb = '+str(np.round(Plist[0][z[0]],2))+', ub = '+str(np.round(Plist[1][z[1]],2)))
#%%
# record the youden's constant, lb, ub, then we can plot the variance of these variables across fish. Also, we should incorporate the DSI student's new 
# filtering methods to see if this makes the automated scoring more accurate.



