# assumes data folder is in src directory named EPM or OF
# must have three other folders in these data folders - image, graph, data

from __future__ import division    
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import my_utilities as myutil
# import seaborn as sns
from scipy.stats import gaussian_kde
import sys
import pandas as pd


counters = myutil.Bunch(frame = 0, noCountour = 0, notInBox = 0, countourTooSmall = 0, multipleContours = 0)

# In original code, average = avg of first frames to produce background from which contours are detected. 
# Rewritten so image bgGray replaces all instances of 'average'
class EPM:
    def __init__(self):
        # EPM box properties 


        # origin_x = 313      # x,y coordinates of top left hand corner of the center square of apparatus  
        # origin_y = 209
        # centerBoxSize = 33    # size of center square of apparatus
        # lengthOfArm = 215     # not critical just needs to be longer than the arms
        # self.EPM_box = myutil.Bunch(x = origin_x, y = origin_y, boxSize = centerBoxSize, armlength = lengthOfArm)

        ## 
        # origin_x = 313      # x,y coordinates of top left hand corner of the center square of apparatus  
        # origin_y = 194
        # centerBoxSize = 31    # size of center square of apparatus
        # lengthOfArm = 215     # not critical just needs to be longer than the arms
        # self.EPM_box = myutil.Bunch(x = origin_x, y = origin_y, boxSize = centerBoxSize, armlength = lengthOfArm)

        ## From Feb 2 videos:
        origin_x = 314      # x,y coordinates of top left hand corner of the center square of apparatus  
        origin_y = 237
        centerBoxSize = 28    # size of center square of apparatus
        lengthOfArm = 180     # not critical just needs to be longer than the arms
        self.EPM_box = myutil.Bunch(x = origin_x, y = origin_y, boxSize = centerBoxSize, armlength = lengthOfArm)

        ## From feb. 5 videos
        # origin_x = 321      # x,y coordinates of top left hand corner of the center square of apparatus  
        # origin_y = 219
        # centerBoxSize = 31    # size of center square of apparatus
        # lengthOfArm = 215     # not critical just needs to be longer than the arms
        # self.EPM_box = myutil.Bunch(x = origin_x, y = origin_y, boxSize = centerBoxSize, armlength = lengthOfArm)

        #Feb 8 vids
        origin_x = 292      # x,y coordinates of top left hand corner of the center square of apparatus  
        origin_y = 216
        centerBoxSize = 28    # size of center square of apparatus
        lengthOfArm = 180     # not critical just needs to be longer than the arms
        self.EPM_box = myutil.Bunch(x = origin_x, y = origin_y, boxSize = centerBoxSize, armlength = lengthOfArm)

        #Feb 13
        origin_x = 290      # x,y coordinates of top left hand corner of the center square of apparatus  
        origin_y = 221
        centerBoxSize = 28    # size of center square of apparatus
        lengthOfArm = 180     # not critical just needs to be longer than the arms
        self.EPM_box = myutil.Bunch(x = origin_x, y = origin_y, boxSize = centerBoxSize, armlength = lengthOfArm)



    def getDimensions(self):
        return self.EPM_box.x, self.EPM_box.y, self.EPM_box.boxSize

    def getBoxes(self):
    # updates the EPM box coordinates
        x = self.EPM_box.x
        y = self.EPM_box.y
        box = self.EPM_box.boxSize
        arm = self.EPM_box.armlength
        boundarybox1 = [(x, y-arm), (x+box, y+box+arm)]   # boundary box 1
        boundarybox2 = [(x-arm, y), (x+box+arm, y+box)]   # boundary box 2
        centerbox = [(x, y), (x+box, y+box)]
        closed_L = [(x-arm, y), (x, y+box)]
        closed_R = [(x+box, y), (x+box+arm, y+box)]
        open_U_near = [(x, y-box), (x+box, y)]
        open_U_far = [(x, y-arm), (x+box, y-box)]
        open_D_near = [(x, y+box), (x+box, y+box+box)]
        open_D_far = [(x, y+box+box), (x+box, y+box+arm)]

        return [boundarybox1, boundarybox2, centerbox, closed_L, closed_R, open_U_near, open_U_far, open_D_near, open_D_far]

    def adjustBoxes(self, bgGray):
    # allow user to reposition/resize the boxes, break from the loop when `c` is pressed
        boxes = self.getBoxes()
        key = cv2.waitKey(1) & 0xFF
        while key != ord("c"):
            # need to continually refresh boxFrame because no other way to clear rectangle
            # boxFrame = myutil.resize(average, width = myutil.video_size)
            boxFrame = bgGray
            boxFrame = myutil.addBoxes(boxFrame, boxes)
            cv2.putText(boxFrame, 'Press \'c\' if box position is OK', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            cv2.imshow(file_name, boxFrame)
            if key == ord("r"): # right arrow
                self.EPM_box.x = self.EPM_box.x + 1
            elif key == ord("l"): # left arrow
                self.EPM_box.x = self.EPM_box.x - 1
            elif key == ord("u"): # up arrow
                self.EPM_box.y = self.EPM_box.y - 1
            elif key == ord("d"): # down arrow
                self.EPM_box.y = self.EPM_box.y + 1
            elif key == ord("."): # down arrow, boxSize has to be a multiple of 4
                self.EPM_box.boxSize = self.EPM_box.boxSize + 1
            elif key == ord(","): # down arrow, boxSize has to be a multiple of 4
                self.EPM_box.boxSize = self.EPM_box.boxSize - 1

            boxes = self.getBoxes()
            key = cv2.waitKey(5) & 0xFF
        #  print box parameters
        print 'x = ', self.EPM_box.x, 'y = ', self.EPM_box.y, 'boxSize = ', self.EPM_box.boxSize
        return boxes

    def whichBox(self, boxedFrame, cx, cy):
    # finds which box the rat is in
        boxes = self.getBoxes()
        centerbox = boxes[2]
        closed_L = boxes[3]
        closed_R = boxes[4]
        open_U_near = boxes[5]
        open_U_far = boxes[6]
        open_D_near = boxes[7]
        open_D_far = boxes[8]
        boxText = ''
        if centerbox[0][0] <= cx <= centerbox[1][0] and centerbox[0][1] <= cy <= centerbox[1][1]:
            cv2.rectangle(boxedFrame, centerbox[0], centerbox[1], (0, 0,255), 1)
            boxText = 'Center' #mid = center box
        elif closed_L[0][0] <= cx <= closed_L[1][0] and closed_L[0][1] <= cy <= closed_L[1][1]:
            cv2.rectangle(boxedFrame, closed_L[0], closed_L[1], (0, 0,255), 1)
            boxText = 'Closed_left' #CL = closed left
        elif closed_R[0][0] <= cx <= closed_R[1][0] and closed_R[0][1] <= cy <= closed_R[1][1]:
            cv2.rectangle(boxedFrame, closed_R[0], closed_R[1], (0, 0,255), 1)
            boxText = 'Closed_right' #CR = closed right
        elif open_U_near[0][0] <= cx <= open_U_near[1][0] and open_U_near[0][1] <= cy <= open_U_near[1][1]:
            cv2.rectangle(boxedFrame, open_U_near[0], open_U_near[1], (0, 0,255), 1)
            boxText = 'Open_up_near' #open up near
        elif open_U_far[0][0] <= cx <= open_U_far[1][0] and open_U_far[0][1] <= cy <= open_U_far[1][1]:
            cv2.rectangle(boxedFrame, open_U_far[0], open_U_far[1], (0, 0,255), 1)
            boxText = 'Open_up_far' #open up far
        elif open_D_near[0][0] <= cx <= open_D_near[1][0] and open_D_near[0][1] <= cy <= open_D_near[1][1]:
            cv2.rectangle(boxedFrame, open_D_near[0], open_D_near[1], (0, 0,255), 1)
            boxText = 'Open_down_near' #open down near
        elif open_D_far[0][0] <= cx <= open_D_far[1][0] and open_D_far[0][1] <= cy <= open_D_far[1][1]:
            cv2.rectangle(boxedFrame, open_D_far[0], open_D_far[1], (0, 0,255), 1)
            boxText = 'Open_down_far' #open down far
        else:
            print 'Error: should not happen'
        return boxedFrame, boxText

    def ratInBox(self, cx, cy):
        boxes = self.getBoxes()
        boundarybox1 = boxes[0]
        boundarybox2 = boxes[1]
        min_x1 = boundarybox1[0][0]
        max_x1 = boundarybox1[1][0]
        min_y1 = boundarybox1[0][1]
        max_y1 = boundarybox1[1][1]
        min_x2 = boundarybox2[0][0]
        max_x2 = boundarybox2[1][0]
        min_y2 = boundarybox2[0][1]
        max_y2 = boundarybox2[1][1]
        inBox = (min_x1 <= cx <= max_x1 and min_y1 <= cy <= max_y1) or (min_x2 <= cx <= max_x2 and min_y2 <= cy <= max_y2)
        return inBox

    def getBoxText(self):
        txt = '\t' + str(self.EPM_box.x) + '\t' + str(self.EPM_box.y) + '\t' + str(self.EPM_box.boxSize) + '\n'
        return txt

    def getPlotLimits(self):
        return ([-300,300],[-200,250])




# NN: Removed original function getBackground() that averages first set of frames to produce background 
# NN: Removed original function skipXseconds()  

def findRat(camera, backgroundFrame):
# find when rat is first alone in box by testing when centroid of largest contour is within the box
    global ratFrame
    frameNumber = 0 
    ratInFrames = 1
    boxes = assaySpecific.getBoxes()
    cnt = []     # stores 'rat' contour

    
    while ratInFrames < 50: # Conditions must be satisfied for 50 consecutive frames
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
                    if assaySpecific.ratInBox(cx, cy):
                        newArea = cv2.contourArea(c)
                        if newArea > oldArea:
                            cnt = c
                            oldArea = newArea
            else: # only 1 potential 'rat'
                cnt = contours[0]

            # if assaySpecific.ratInBox(cx, cy):
            #     continue
            # else:
            #     ratInFrames=0

            if cnt == []:  # deal with case where there are multiple contours but none in box so that cnt is not defined in first frame
                pass
            else:
                # find centroid
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                # test if 'rat' is in box and is big enough
                if assaySpecific.ratInBox(cx, cy) and cv2.contourArea(cnt) > myutil.min_area and len(contours) < 7:
                    ratInFrames += 1 
                    ratFrame = frameNumber

                    #print "Frame %s: "%frameNumber + "mouse found."
                else:
                    ratInFrames = 0
                    #print "Frame %s: "%frameNumber + str(len(contours))+" contours" # For debugging purposes - must have <7 contours for 50 straight frames to start tracking

        if Single:
            # frame = myutil.addBoxes(frame, boxes)
            cv2.imshow(file_name, frame)
            # if the `q` key is pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


        frameNumber += 1



# Added function so no countours will be analyzed from the top right hand corner where the time in HH:MM:SS has been annotated onto each video 
def excludeTimecode(c):
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    if cx > timecode_x and cy < timecode_y:
        return True

# def excludeTopright(c):
#     M = cv2.moments(c)
#     cx = int(M['m10']/M['m00'])
#     cy = int(M['m01']/M['m00'])
#     if cx > topright_x and cy < topright_y:
#         return True

def trackRat(camera, backgroundFrame, startFrame):
# track rat during the rest of the video
    boxes = assaySpecific.getBoxes()
    # storage for rat location
    xList = []
    yList = []
    dataText = ''
    
# reset counters
    counters.frame = 0
    counters.noCountour = 0
    counters.countourTooSmall = 0
    counters.notInBox = 0
    counters.multipleContours = 0
    cx = 0  # initialize these coordinates to deal with the problem with the first frame
    cy = 0
    box_text = ''
    
    camera.set(cv2.CAP_PROP_POS_FRAMES, startFrame) # Set the frame to startFrame
    (grabbed, frame) = camera.read() # grab first frame in analysis period #boolean 


    # perform analysis for 300 seconds (5 minutes)
    while camera.isOpened() and grabbed and (counters.frame <= myutil.fps * 5 * 60):
        frame, contours = myutil.findCountours(frame, backgroundFrame)

        if exclude_timecode:
            contours = filter(lambda c: not excludeTimecode(c), contours)

        # if exclude_topright:
        #     contours = filter(lambda c: not excludeTopright(c), contours)

        if not contours: # no rat
            #print counters.frame, '  Warning: no contours'
            counters.noCountour = counters.noCountour + 1
            # if there are no contours then program will use the previous frames values, cnt is unchanged,
            # this fails if there no contours in the first frame to be analyzed, in which case the initial values are used,
            # which will typically fall outside box
        else:
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
                    if assaySpecific.ratInBox(cx, cy) and cv2.contourArea(c) > myutil.min_area:
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

            # find centroid location
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # check if 'rat' is big enough
            if cv2.contourArea(cnt) < myutil.min_area:
                print counters.frame, '  Warning: contour too small', '  area = ', cv2.contourArea(cnt)
                counters.countourTooSmall = counters.countourTooSmall + 1


        # update seconds elapsed counter on screen
        elapsedSeconds = int(round((camera.get(cv2.CAP_PROP_POS_FRAMES)-startFrame)/myutil.fps))
        cv2.putText(frame, str(elapsedSeconds), (frame.shape[1] - 65, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        

        frame = myutil.addBoxes(frame, boxes) # add boxes after checking for contours

        # if 'rat' is in box, find which box, add marker and store data
        # in the case of contours = null, previous frames values are used as best estimate
        if assaySpecific.ratInBox(cx, cy):
            frame, box_text = assaySpecific.whichBox(frame, cx, cy)     # this adds red box to image
            frame = cv2.circle(frame, (cx,cy), 2, (0,0,255), 3)     # add red dot
            dataText = dataText + str(counters.frame) + '\t' + str(cx) + '\t' + str(cy) + '\t' + box_text + '\n'
            xList.append(cx)
            yList.append(cy)
        else:
            print counters.frame, 'Warning: mouse not in box'
            counters.notInBox = counters.notInBox + 1

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

    return dataText, xList, yList, lastFrame, lastFrameNumber

def printToConsole(counters, lastFrameNumber):
# save data file and print Warning messages to console
    print 'Warning: There were ', counters.noCountour, '(', str('%.1f' %(counters.noCountour/counters.frame*100)), '%) no contour warnings.'
    print 'Warning: There were ', counters.notInBox, '(', str('%.1f' %(counters.notInBox/counters.frame*100)), '%) not in the box warnings.'
    print 'Warning: There were ', counters.countourTooSmall, '(', str('%.1f' %(counters.countourTooSmall/counters.frame*100)), '%) contour too small warnings.'
    print 'Warning: There were ', counters.multipleContours, '(', str('%.1f' %(counters.multipleContours/counters.frame*100)), '%) occurrences of multiple contours.'

    print 'Total number of frames processed = ', counters.frame-1
    print 'Last frame number = ', int(lastFrameNumber)
    secondsProcessed = (lastFrameNumber)/myutil.fps
    print 'Seconds of video processed = ', secondsProcessed

def saveData(headerText, dataText, lastFrame, lastFrameNumber):
    headerText = headerText + 'Warning: There were ' + str(counters.noCountour) +  '(' + str('%.1f' %(counters.noCountour/counters.frame*100)) + '%) no contour warnings.' + '\n'
    headerText = headerText + 'Warning: There were ' + str(counters.notInBox) +  '(' + str('%.1f' %(counters.notInBox/counters.frame*100)) + '%) not in the box warnings.' + '\n'
    headerText = headerText + 'Warning: There were ' + str(counters.countourTooSmall) +  '(' + str('%.1f' %(counters.countourTooSmall/counters.frame*100)) + '%) contour too small warnings.' + '\n'
    headerText = headerText + 'Warning: There were ' + str(counters.multipleContours) +  '(' + str('%.1f' %(counters.multipleContours/counters.frame*100)) + '%) occurrences of multiple contours.' + '\n'
    headerText = headerText + 'Total number of frames processed = ' + str(counters.frame-1) + '\n'
    secondsProcessed = (lastFrameNumber)/myutil.fps
    headerText = headerText + 'Seconds of video processed = ' + str(secondsProcessed) + '\n'
    assayText = assaySpecific.getBoxText()
    headerText = headerText + str(counters.frame-1) + '\t' + str(counters.noCountour) + '\t' + str(counters.notInBox) + '\t' + str(counters.countourTooSmall) + '\t' + str(counters.multipleContours) + assayText

    # save data file
    with open(base_Path + 'data/' + file_name + '.txt', 'w') as f:
        f.write(headerText + dataText)
    f.close()

    # save last image to file, can check this to see if everything looks OK
    cv2.imwrite(base_Path + 'image/' + file_name + '.png', lastFrame)

def plotData(xList, yList):
# plot data and save file
    x, y, box = assaySpecific.getDimensions()
    plot_x = np.array(xList) - x
    plot_y = (np.array(yList) - y)*-1 + box   # invert y-axis to match video orientation
    # plotlimits = assaySpecific.getPlotLimits()  # get assay specific plot limits
    # plt.xlim(plotlimits[0])
    # plt.ylim(plotlimits[1])

    # plt.plot(plot_x, plot_y)
    # hm = heatmap.Heatmap()
    # img = hm.heatmap((plot_x,plot_y)).save("test_heatmap.png")
    # heatmap, xedges, yedges = np.histogram2d(plot_x, plot_y, bins=20)
    # plt.clf()
    # plt.imshow(heatmap.T, origin='lower')
    # plt.show()

    # sns.set(style="white", color_codes=True, palette=1)
    # sns.palplot(sns.color_palette("RdBu", n_colors=7))
    # sns.jointplot(x=plot_x, y=plot_y, kind='kde', color="skyblue")
    # sns.jointplot(x=plot_x, y=plot_y, kind="hex", stat_func=None)

    # sns.jointplot(x=plot_x, y=plot_y, color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6)
    # sns.kdeplot(x=plot_x, y=plot_y, cmap="Reds", shade=True, shade_lowest=False)

    a=plot_x
    b=plot_y
    # Calculate the point density
    ab = np.vstack([plot_x,plot_y])
    z = gaussian_kde(ab)(ab)
    # # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    plot_x, plot_y, z = plot_x[idx], plot_y[idx], z[idx]

    # NN: Create heatmap
    fig, ax = plt.subplots()
    ax.scatter(plot_x, plot_y, c=z, s=50, cmap=plt.cm.jet, edgecolor='')
    plt.ylim([-200,200])
    plt.xlim([-200,200])
    plt.savefig(base_Path + 'graph/' + file_name + '.png', bbox_inches='tight')  
    plt.close()

    # Test heatmap
    # sns.set()
    # normal_data = np.random.randn(10, 12)
    # heatmap = sns.heatmap(normal_data, center=0)
    # heatmap.savefig("output.png")

    # Mick's Heatmap
    # df = pandas.DataFrame({"x" : plot_x, "y" : plot_y, "z":z})
    # res = df.groupby(['y','x'])['z'].mean().unstack()
    # ax = sns.heatmap(res)
    # ax.axis('equal')
    # ax.invert_yaxis()
    # ax = sns.heatmap(df,cmap='RdBu')
    # fig = ax.get_figure()
    # fig.savefig('./out.png')

    # df['Z_value'] = pd.to_numeric(df['Z_value'])

    # uniform_data = np.random.rand(10, 12)
    # ax = sns.heatmap(uniform_data)

    # flights = sns.load_dataset("flights")
    # flights = flights.pivot("x", "y", "frames")
    # ax = sns.heatmap(flights)
    # ax = sns.heatmap(flights, annot=True, fmt="d")


    # plt.hist2d(a, b, (100, 100), cmap=plt.cm.Greys)
    # plt.colorbar()
    # plt.ylim([-175,175])
    # plt.xlim([-175,175])
    # plt.show()
    # plt.savefig(base_Path + 'graph/' + file_name + '.png', bbox_inches='tight')
    # plt.close()
    
    # plt.savefig(file_name + '.pdf', bbox_inches='tight')    # print pdf
    # if Single:  # only show plot if single file
    #     plt.show()
    # plt.close()    # destroy fig so that next cycle does not overwrite this one

def processFile(base_Path, file_name, testType):
    
    noRat = True
    camera = cv2.VideoCapture(base_Path+file_name)      # open video file

    # NN: Next lines to replace getBackground function  
    imageFolder = './EPM/backgroundImage/'
    date = file_name.split('_')[0]
    if os.path.exists(imageFolder + file_name + '.jpg'):
        image = imageFolder + file_name + '.jpg' 
    else:
        image = imageFolder + date + '.jpg'
    bgFrame = cv2.imread(image) # NN: reads image within backgroundImage folder
    bgGray = cv2.cvtColor(bgFrame, cv2.COLOR_BGR2GRAY) # NN: convert to grayscale
    backgroundFrame = cv2.GaussianBlur(bgGray, (myutil.ksize, myutil.ksize), 0)  ## NN: Blurs image

    print ('Running video file ' + file_name + ' using background image ' + image )
    headerText = file_name + '\n'

    if Single:  # user interaction to adjust box size, only for single files
        assaySpecific.adjustBoxes(bgGray)
    findRat(camera, backgroundFrame)       # advances through file until there is a 'rat' within the boundary boxes
    print("Started tracking at frame " +str(ratFrame) + " (skipped "+str(int(ratFrame/30))+" seconds)")
    dataText, xList, yList, lastFrame, lastFrameNumber = trackRat(camera, backgroundFrame, ratFrame)  # track rat for 5 minutes
    printToConsole(counters, lastFrameNumber)        # print quality control stats
    saveData(headerText, dataText, lastFrame, lastFrameNumber)  # save data to file
    plotData(xList, yList) 

    # cleanup the camera and close any open windows
    camera.release() # Close the webcam
    cv2.destroyAllWindows()


base_Path = './EPM/'
#file_name = '20180216_PO_M_144.mp4' #if multiple then comment this out
testType = 'EPM'

if testType == 'EPM':
    assaySpecific = EPM()

exclude_timecode = True
# exclude_topright = True
timecode_x = 360
timecode_y = 48

# topright_x = 150
# topright_y = 48

Single = False # True if putting in only one video
Multiple =  not Single

if Single:
    processFile(base_Path, file_name, testType)

# if multiple, iterate through videos and corresponding background images
elif Multiple:
    fileList = os.listdir(base_Path)
    for file_name in fileList:
        path = os.path.join(base_Path, file_name)
        if not os.path.isdir(path) and not file_name.startswith('.'): # NN: skip directories and dotfiles
            processFile(base_Path, file_name, testType)