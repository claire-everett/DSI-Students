#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:11:28 2021

@author: claireeverett
"""
import cv2
import numpy as np
from helper_functions_VA import *
import os
from glob import glob
import re
def manual_scoring(data_manual, crop0 = 72000,crop1= 144000):
    '''
    A function that takes manually scored data and converts it to a binary array. 
    
    Parameters: 
    data_manual: manual scored data, read in from an excel file
    data_auto: automatically scored data, just used to establish how long the session is. 
    
    Returns: 
    pandas array: binary array of open/closed scoring
    '''
    
    Manual = pd.DataFrame(0, index=np.arange(crop0,crop1), columns = ['OpOpen'])
    reference = data_manual.index
    if len(reference) > 0: 
        for i in reference:
            Manual[data_manual['Start'][i]-crop0:data_manual['Stop'][i]-crop0] = 1
        
        print(Manual[data_manual['Start'][i]:data_manual['Stop'][i]])  
        return Manual['OpOpen']
    else:
        return Manual['OpOpen']
        
#%%
#variable definitions
import_dir = '/Users/claireeverett/Desktop/For_Andres/training/h5'
export_dir = '/Users/claireeverett/Desktop/For_Andres/training/positive'
# auto data
h5_files = sorted(glob(os.path.join(import_dir,'*.h5')))
'''CHANGE'''
for i in np.arange(len(h5_files)):
    file_handle_auto = h5_files[11]
    with pd.HDFStore(file_handle_auto,'r') as file:
        data_auto = file.get('df_with_missing')
        data_auto.columns= data_auto.columns.droplevel()
    data_auto['Midpointx'], data_auto['Midpointy'] = midpointx(data_auto['B_rightoperculum']['x'], data_auto['B_rightoperculum']['y'], data_auto['E_leftoperculum']['x'], data_auto['E_leftoperculum']['y'])

# new_features=features(starttime=0, duration=len(data_auto))
    
# filtered_df=new_features.filter_df(data_auto, add_midpoint = True)

# new_features.fit(filtered_df,filter_feature=True,fill_na=True,estimate_na=False)
    
#%%

'''CHANGE'''
import_dir_xlsx = '/Users/claireeverett/Desktop/For_Andres/training/xlsx'

# manual 
excel_files = sorted(glob(os.path.join(import_dir_xlsx,'*.xlsx')))
for i in np.arange(len(excel_files)):
    file_handle = excel_files[11]
    data_manual = pd.read_excel(file_handle)

data_manual['Start'] = data_manual['Start'].astype(int)
data_manual['Stop'] = data_manual['Stop'].astype(int)

manual_scoring_ex = manual_scoring(data_manual,0,len(data_auto)-1)


#%%

ref_x = data_auto['A_head']['x']
ref_y = data_auto['A_head']['y']
#%%

# create list of all open frames
frameIds = []
for j in np.arange(len(manual_scoring_ex)):
    if manual_scoring_ex.values[j] == 1:
        frameIds.append(j)

# frameIds = frameIds[:50]

frameIds_pd = pd.DataFrame(frameIds)
frameIds_adjust = [x -0 for x in frameIds]
# frameIds_pd.to_excel(os.path.join(export_dir, 'FP'+str(os.path.basename(h5_files[i]).split('_')[0]) + str(os.path.basename(h5_files[i]).split('_')[2]) + "frameIds.xlsx"))

#%%

frameIds_adjust = frameIds_adjust[::20]

#%%
# make video of positive examples
import_dir_vid = '/Users/claireeverett/Desktop/For_Andres/training/mp4'

vid_files = sorted(glob(os.path.join(import_dir_vid,'*.mp4')))

'''CHANGE'''
vid_file_og = vid_files[11]
print(vid_file_og)

# Make video of all the false positives
cap = cv2.VideoCapture(os.path.join(import_dir, vid_file_og))

frames = []
count = 0
for fid in frameIds_adjust:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
    
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(os.path.join(export_dir, str(os.path.basename(vid_file_og).split('_')[:4]) +'.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 40, (frame_width,frame_height))
# out = cv2.VideoWriter(os.path.join(import_dir, 'Compare.mp4'),fourcc, 40.0, (frame_width,frame_height))

for fid in frames:
    out.write(fid)
    

#%%
#convert video to image sequence
import_dir_avi = '/Users/claireeverett/Desktop/For_Andres/training/positive'
import_dir_jpeg = '/Users/claireeverett/Desktop/For_Andres/training/positive/full_frame'
vid_files = sorted(glob(os.path.join(import_dir_avi,'*.avi')))

vid_file = vid_files[0]


vidcap = cv2.VideoCapture(vid_file)

success,image = vidcap.read()

count = 0
while success:
  cv2.imwrite(os.path.join(import_dir_jpeg,os.path.basename(vid_file_og).split('_')[0]+os.path.basename(vid_file_og).split('_')[1]+ os.path.basename(vid_file_og).split('_')[2]+ os.path.basename(vid_file_og).split('_')[3][0] + '_'+ "%d.jpeg" % frameIds_adjust[count]), image)     # save frame as JPEG file     
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count = count + 1


#%%
frameIds_adjust = [x + 1 for x in frameIds_adjust]
#%%
#load the images and mask them 

import_dir_mask = '/Users/claireeverett/Desktop/For_Andres/training/positive/mask'
count = 0
circle_radius = 50
jpg_files =  sorted(glob(os.path.join(import_dir_jpeg,'*.jpeg')))

jpg_files.sort(key=lambda f: int(re.sub('\D', '', f)))

count = 0
for i in np.arange(len(jpg_files)):
    jpg_file = jpg_files[i]
    print(jpg_file)
    image = cv2.imread(jpg_file) 
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = cv2.circle(mask, (int(ref_x[frameIds_adjust[i]]), int(ref_y[frameIds_adjust[i]])), circle_radius, (255,255,255), -1) 
    
    # Mask input image with binary mask
    result = cv2.bitwise_and(image, mask)
    # Color background white
    result[mask==0] = 255 # Optional
    # crop_result = result[int(ref_x[frameIds[i]]):65, int(ref_y[frameIds[i]]):65]
    crop_result = result[int(ref_y[frameIds_adjust[i]])-circle_radius:int(ref_y[frameIds_adjust[i]])+circle_radius, int(ref_x[frameIds_adjust[i]])-circle_radius:int(ref_x[frameIds_adjust[i]])+circle_radius]
    # cv2.imshow('cropped', crop_result)
    '''
    im currently working on cropping so I've commented out the imwrite of the results and the showing of the reults
    it's difficult right now bc when I try to crop using correct coordinates, it tells me I am out of space. I need to reduce
    the number of images I cycle through to trouble shoot faster'
    '''
    # cv2.imshow('image', image)
    # cv2.imshow('mask', mask)
    # cv2.imshow('result', result)
    
    # cv2.waitKey()
    cv2.imwrite(os.path.join(import_dir_mask,os.path.basename(vid_file_og).split('_')[0]+os.path.basename(vid_file_og).split('_')[1]+ os.path.basename(vid_file_og).split('_')[2]+ os.path.basename(vid_file_og).split('_')[3][0] +'_'+"%d.jpeg" % frameIds_adjust[count]), crop_result)
    count = count + 1
              
#%%
vid_files = sorted(glob(os.path.join(import_dir,'*.mp4')))


vid_file = vid_files[0]

# Make video of all the false positives
cap = cv2.VideoCapture(os.path.join(import_dir, vid_file))

frames = []

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = cv2.VideoWriter(os.path.join(import_dir, 'FN' + '_' + str(os.path.basename(vid_file).split('_')[2]) + str(os.path.basename(vid_file).split('_')[0])) + 'Compare.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

for fid in frames:
    out.write(fid)
    
    
#%%
#rename the files to be better suited for later manipulation
# import_dir_jpeg = '/Users/claireeverett/Desktop/For_Andres/training/positive/full_frame_final'
# jpg_files =  sorted(glob(os.path.join(import_dir_jpeg,'*.jpeg')))
# jpg_files.sort(key=lambda f: int(re.sub('\D', '', f)))

# jpg_file = jpg_files[0]
# new_name_list = []
# for i in np.arange(len(os.path.basename(jpg_file).split(','))):
#     for j in np.arange(len(os.path.basename(jpg_file).split(',')[i])):
#         if os.path.basename(jpg_file).split(',')[i][j].isalpha():
#             print(os.path.basename(jpg_file).split(',')[i][j])
#             new_name_list.append(os.path.basename(jpg_file).split(',')[i][j])
        
# new_name = str()
    
    