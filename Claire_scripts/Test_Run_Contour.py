#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:11:59 2020

@author: Claire
"""
# import packages
from __future__ import division    
import cv2
import my_utilities as myutil
#%%
# Define teh background image, the video to use

base_Path = './'
file_name = 'Reverse_888.mp4' #if multiple then comment this out
Single = True

camera = cv2.VideoCapture(base_Path+file_name) 
image = 'test.png'

bgFrame = cv2.imread(image) # NN: reads image within backgroundImage folder
bgGray = cv2.cvtColor(bgFrame, cv2.COLOR_BGR2GRAY) # NN: convert to grayscale
backgroundFrame = cv2.GaussianBlur(bgGray, (100, 100), 0)  ## NN: Blurs image

#%%
# create a greyscale using the background image, makes a video

frameNumber = 0
cnt = []
ContoursInFrames = 1

while ContoursInFrames < 50:
    (_, frame) = camera.read()
    frame, contours = myutil.findCountours(frame, backgroundFrame)
    
    if contours == []: # no 'rat'
            pass
    else:
            if len(contours) > 1: # more than 1 potential 'rat'
                # find largest contour with its centroid in the box in the hope that this is the rat
                oldArea = 0
                for c in contours:
                    # find centroid
                    M = cv2.moments(c)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    newArea = cv2.contourArea(c)
                    if newArea > oldArea:
                        cnt = c
                        oldArea = newArea
            else: # only 1 potential 'rat'
                cnt = contours[0]
                
            if cnt == []:  # deal with case where there are multiple contours but none in box so that cnt is not defined in first frame
                pass
            else:
                # find centroid
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                # # test if 'rat' is in box and is big enough
                # if assaySpecific.ratInBox(cx, cy) and cv2.contourArea(cnt) > myutil.min_area and len(contours) < 7:
                #     ratInFrames += 1 
                #     ratFrame = frameNumber

                #     #print "Frame %s: "%frameNumber + "mouse found."
                # else:
                #     ratInFrames = 0
                #     #print "Frame %s: "%frameNumber + str(len(contours))+" contours" # For debugging purposes - must have <7 contours for 50 straight frames to start tracking

            if Single:
                # frame = myutil.addBoxes(frame, boxes)
                cv2.imshow(file_name, frame)
                # if the `q` key is pressed, break from the loop
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break


            frameNumber += 1
    print(cnt)
#%%
# use the greyscale image to create a contour
counters = myutil.Bunch(frame = 0, noCountour = 0, notInBox = 0, countourTooSmall = 0, multipleContours = 0)

startFrame = 0

xList = []
yList = []
dataText = ''
    
counters.frame = 0
counters.noCountour = 0
counters.countourTooSmall = 0
counters.notInBox = 0
counters.multipleContours = 0
cx = 0  # initialize these coordinates to deal with the problem with the first frame
cy = 0   

camera.set(cv2.CAP_PROP_POS_FRAMES, startFrame) # Set the frame to startFrame
(grabbed, frame) = camera.read() # grab first frame in analysis period #boolean 


# perform analysis for 300 seconds (5 minutes)
while camera.isOpened() and grabbed and (counters.frame <= myutil.fps * 5 * 60):
    frame, contours = myutil.findCountours(frame, backgroundFrame)
    frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow(file_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break    
    lastFrameNumber = camera.get(cv2.CAP_PROP_POS_FRAMES)-startFrame
    lastFrame = frame
    counters.frame = counters.frame + 1
    (grabbed, frame) = camera.read()

#%%    
    
    if len(contours) > 1: # more than 1 potential 'rat'
        counters.multipleContours = counters.multipleContours + 1
        # find largest contour with its centroid in the box in the hope that this is the rat
        # this fails if none of the contours in the first frame fall inside box and/or are big enough
        oldArea = 0

        for c in contours:
            # find centroid
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # test if 'rat' is in box and is big enough
            if cv2.contourArea(c) > myutil.min_area:
                if cv2.contourArea(c) < myutil.max_area:
                    newArea = cv2.contourArea(c)
                    if newArea > oldArea:
                        cnt = c
                        oldArea = newArea
            else:
                counters.noCountour = counters.noCountour + 1
                cnt = c
    else: # only 1 'rat'
        cnt = contours[0]

    # Show all contours
    frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    if Single:
        cv2.imshow(file_name, frame)
        # if the `q` key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    lastFrameNumber = camera.get(cv2.CAP_PROP_POS_FRAMES)-startFrame
    lastFrame = frame
    counters.frame = counters.frame + 1
    (grabbed, frame) = camera.read() # get next frame
#%%
#draw contours on the video
