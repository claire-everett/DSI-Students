#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:24:45 2021

@author: ryan
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import h5py


from contour_utils import find_largest_contour,find_centroid
#from tqdm import tqdm

CROP_LENGTH = 200
FPS = 40
def my_window(image,head_x,head_y, crop_length = 200):
    l = image.shape[0]
    w = image.shape[1]
    half_crop = crop_length//2
    if head_x + half_crop >= w:
        w_max = w
        w_min = w - crop_length
    elif head_x - half_crop < 0:
        w_max = crop_length
        w_min = 0 
    else:
        w_max = head_x + half_crop
        w_min = head_x - half_crop
    if head_y + half_crop >= l:
        l_max = l
        l_min = l-crop_length
    elif head_y - half_crop < 0:
        l_max = crop_length
        l_min = 0
    else:
        l_max = head_y + half_crop
        l_min = head_y - half_crop
    return image[l_min:l_max,w_min:w_max]
def find_background(video_path,sample_length = None,image_shape = (500,500,3)):
    '''
    sample_length: number of samples used to calculate background, default to full video
    '''
    if not os.path.exists(video_path):
        raise Exception("video path not exist")
    cap = cv2.VideoCapture(video_path) 
    frames = np.zeros(image_shape) #running mean of video
    if sample_length is None:
        sample_length = float("inf")
    n_sample = 0
    while True:
        ret, frame = cap.read() 
        n_sample += 1
        if not ret or n_sample>sample_length:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames = frames*(n_sample-1)/n_sample + 1/n_sample*frame #memory efficient, time is roughly same as keeping a list
        if n_sample % 5000 == 0:
            print("{} samples processed".format(n_sample))
    #frames = np.stack(frames)
    #background = np.mean(frames,axis = 0)
    print("background for {}".format(video_path))
    plt.figure()
    plt.imshow(np.array(frames,dtype = np.uint8))
    plt.show()
    return frames

def remove_centralize(video_path,bg,crop_length = 200, output_path = None):
    if not os.path.exists(video_path):
        raise Exception("video path not exist")
    if output_path is None:
        output_path = ".".join(video_path.split(".")[:-1])+"_truncated.wav"
    cap = cv2.VideoCapture(video_path) 
    result = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'), FPS, (CROP_LENGTH,CROP_LENGTH))
    index = 0
    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        new_frame = cv2.subtract(np.invert(np.array(frame,dtype = np.uint8)),np.invert(np.array(bg,dtype = np.uint8)))
        #invert here is because fish body is usually back, which makes direct
        #subtract a black image
        #new_frame = np.invert(new_frame)
        gray = cv2.cvtColor(new_frame,cv2.COLOR_RGB2GRAY)
        ret,th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fish_contour,flag=find_largest_contour(contours)
        '''
        if flag==0:
            print("no valid fish contour find at index {}".format(index))
            img=np.float32(np.full(frame.shape,0))#just in case there's no valid contour, won't happen in the current case
        else:
            #draw only the mask of this contour
            img=cv2.drawContours(np.float32(np.full(frame.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        '''
        xbar,ybar = find_centroid(fish_contour)
        cropped_frame = my_window(frame,xbar,ybar,CROP_LENGTH)
        result.write(cropped_frame)
        index += 1
    result.release()
    return output_path

def make_training_h5(cropped_video_path,start,stop):
    #This just put frames from video into a h5
    fish_name = "_".join(os.path.basename(cropped_video_path).split("_")[:-1])
    outputs = []
    if not os.path.exists(cropped_video_path):
        raise Exception("video path not exist")
    index = start
    cap = cv2.VideoCapture(video_path) 
    cap.set(1,index)
    while True:
        ret,frame = cap.read()
        if not ret or index>=stop:
            break
        outputs.append(frame)
        index += 1
    outputs = np.stack(outputs)
    file = h5py.File(os.path.join(os.path.dirname(cropped_video_path),fish_name+".h5"), "w")
    file.create_dataset(
        "images", np.shape(outputs), h5py.h5t.STD_U8BE, data=outputs)
    file.create_dataset(
        'start',data = start)
    file.create_dataset(
        "stop",data = stop)
    file.close()
    print("finished getting data for {}".format(fish_name))
    
video_path = "/Users/ryan/Desktop/FishProject/Scripts/TailBeatingExamples/PS_1.1.2_L_CE.mp4"
start = 90000
stop = 140000
def main():
    bg = find_background(video_path,sample_length = 50000)
    cropped_video_path = remove_centralize(video_path,bg)
    make_training_h5(cropped_video_path,start,stop)