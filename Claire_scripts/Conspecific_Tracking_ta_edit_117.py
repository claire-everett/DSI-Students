# coding: utf-8

# In[ ]:


#Notebook to quantify the duration of time the fish spends
#following the other fish
#going to measure using the orientation of the fish and how well
# the orientation matches the line from the tip of the head to the tip of the head
#of the other fish

##actually I can do it by just measuring a point in the middle
# of the fish, the tip of th ehead, and the tip of the other fish's head
# then the closer that angle is to 180, the more the fish is "facing" the other fish


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.spatial import distance
import seaborn as sns; sns.set()
import math
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, peak_widths
from scipy.stats import norm
from tqdm import tqdm_notebook as tqdm
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


# ##### Functions

# In[24]:


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

def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    return (data['x'],data['y'])

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

def gaze_ethoplot(anglearrays,title,show = True,save = False):
    """
    Function to generate a plot of the gaze of both fish, and points when they are looking in the same direction. 

    Parameters: 
    anglearrays: a list of two array-likes: one for fish one, one for fish two. 
    title: a string of the title for the plot. 
    show: a boolean. True = plot, False = do not plot
    save: a boolean: True = save, False = do not save. 
    """
    colors = ['red','blue']
    labels = ['fish1 gaze', 'fish2 gaze']
    fig,ax = plt.subplots(2,1,sharex = True)
    for i in range(2):
        anglearray = anglearrays[i]
        inds = np.arange(len(anglearray))
        color = colors[i]
        ax[0].plot(anglearray,color = color,linewidth = 1,alpha = 0.5,label = labels[i])
        boolean = anglearray.apply(lambda x: 1 if x > 140 else 0) 
        [ax[1].axvline(x = j,alpha = 0.2,color = color) for j in inds if boolean[j] == 1]

    ax[0].set_ylabel('relative angle (degrees)')
    ax[1].set_ylabel('threshold crossing')
    ax[1].set_xlabel('time (frame)')
    ax[0].set_title(title)

    ax[0].legend(loc = 1)## in the upper right; faster than best. 

    if save == True:
        plt.savefig(title + '.png')
        
    if show == True:
        plt.show()

def binarize(anglearrays):
    booleans = []
    for i in range(2):
        anglearray = anglearrays[i]
        inds = np.arange(len(anglearray))
        boolean = anglearray.apply(lambda x: 1 if x > 140 else 0).values
        boolean = binary_dilation(boolean, structure = np.ones(40,))
        booleans.append(boolean)
    
    product = booleans[0] * booleans[1]
    result = np.where(product)
    
    return (result)

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
          
    return Manual['OpOpen'][crop0:crop1]

def auto_scoring_TS1(data_auto,thresh_param0 = 65,thresh_param1 = 135.56):
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

def Percent_Trial(func, data_auto):
    BinaryVector = func(data_auto)
    OpOpen = (BinaryVector.sum())/len(BinaryVector)
    return OpOpen

cross = []
index = []

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

def findcrossingunder (Operangle, threshold):
    
    for i in np.arange(len(Operangle)):
        
        if Operangle.values[i-1] > threshold:
            if Operangle.values[i-2] > threshold:
                if Operangle.values[i-3] > threshold:
                    if Operangle.values[i-4] > threshold:
                        if Operangle.values[i-5] > threshold:
                            if Operangle.values[i-6] > threshold:
                                if Operangle.values[i-7] > threshold:
                    
                                    if Operangle.values[i] < threshold:
                                        if Operangle.values[i+1] < threshold:
                                            if Operangle.values[i+2] < threshold:
                                                if Operangle.values[i+3] < threshold:
                                                    if Operangle.values[i+4] < threshold:
                                                        if Operangle.values[i+5] < threshold:
                                                            if Operangle.values[i+6] < threshold:
                                                                if Operangle.values[i+7] < threshold:
                                                                    print(i)
                                                                    cross.append(Operangle.values[i])
                                                                    index.append(i)

    return(cross, index)
    
def findcrossingover (Operangle, threshold):
    
    for i in np.arange(len(Operangle)):
        
        if Operangle.values[i-1] < threshold:
            if Operangle.values[i-2] < threshold:
                if Operangle.values[i-3] < threshold:
                    if Operangle.values[i-4] < threshold:
                        if Operangle.values[i-5] < threshold:
                            if Operangle.values[i-6] < threshold:
                                if Operangle.values[i-7] < threshold:
                    
                                    if Operangle.values[i] > threshold:
                                        if Operangle.values[i+1] > threshold:
                                            if Operangle.values[i+2] > threshold:
                                                if Operangle.values[i+3] > threshold:
                                                    if Operangle.values[i+4] > threshold:
                                                        if Operangle.values[i+5] > threshold:
                                                            if Operangle.values[i+6] > threshold:
                                                                if Operangle.values[i+7] > threshold:
                                                                    print(i)
                                                                    cross.append(Operangle.values[i])
                                                                    index.append(i)

    return(cross, index)
    


def binarizeOp(Operangle, threshold):
    

    boolean = Operangle.apply(lambda x: 1 if x > threshold else 0).values
    boolean = binary_dilation(boolean, structure = np.ones(10,))
    
    binindex = np.where(boolean)[0]
#    for i,value in enumerate(boolean):
#        if value == 1:
#            binindex.append([i])
#            print('hey')
#        else:
#            binindex.append(np.nan)
#            print('bye')
#   
    return (binindex)

def uniform(indbin,LoopLength):
    A = len(indbin)
    B = LoopLength
    C = B - A
    end = np.zeros(C)
    new = np.concatenate((indbin, end))
    
    return(new)
####################################

    ## ##### Create Midpoint of Operculi

    ## In[25]:


    #data_auto1['Midpointx'], data_auto1['Midpointy'] = midpointx(data_auto1['B_rightoperculum']['x'], data_auto1['B_rightoperculum']['y'], data_auto1['E_leftoperculum']['y'], data_auto1['E_leftoperculum']['y'] )


    ## ##### Measure the String Between Fish1 Orientation and Fish2

    ## In[36]:


    ##line 1: between midpoint of op1 and tip of fish head 1
    ## line 2: between tip of fish head 1 and 2
    ## line 3: between tip of fish head 2 and midpoint of op1

    #line1 = mydistance((data_auto1['Midpointx'], data_auto1['Midpointy']), (data_auto1['A_head']['x'], data_auto1['A_head']['y']))
    #line2 = mydistance((data_auto1['A_head']['x'], data_auto1['A_head']['y']), (data_auto2['A_head']['x'], data_auto2['A_head']['y']))
    #line3 = mydistance((data_auto2['A_head']['x'], data_auto2['A_head']['y']), (data_auto1['Midpointx'], data_auto1['Midpointy']))

    #String = lawofcosines(line1, line2, line3)
    #funcstring = gaze_tracking(data_auto2,data_auto1) 
    #plt.plot(funcstring)
    #plt.show()


    ## In[49]:


    #StraightString = String.apply(lambda x: 1 if x > 170 else 0)


    ## In[50]:


    #StraightString.sum()


    # In[53]:


## Functions added by Claire 8/28
def HeatmapCompare (filename, angle, data_manual, start, stop):
    Fish1man = manual_scoring(data_manual, angle[start:stop])
    Fish1aut = angle.apply(lambda x: 1 if x > 65.5 else 0).values
    Compare = pd.DataFrame(0, index=np.arange(len(Fish1man)), columns = ['Manual', 'Automatic'])
    Compare['Manual'] = Fish1man
    Compare['Automatic'] = Fish1aut[start:(stop - 1)]
    ax = sns.heatmap(Compare)
    plt.savefig('heatmapcompare' + str(filename) + '.png')

def DualAngleHeatMapCompare (filename, angle1, angle2, data_manual1, data_manual2, start, stop):
    Fish1man = manual_scoring(data_manual1, angle1[start:stop])
    Fish1aut = angle1.apply(lambda x: 1 if x > 140 else 0).values
    Fish2aut = angle2.apply(lambda x: 1 if x > 140 else 0).values
    Fish2man = manual_scoring(data_manual2, data_auto2[start:stop])
    Compare = pd.DataFrame(0, index=np.arange(len(Fish1man)), columns = ['Manual', 'Automatic'])
    Compare['Manual'] = Fish1man + Fish2man
    Compare['Automatic'] = Fish1aut[start:(stop - 1)] + Fish2aut[start:(stop - 1)]
    ax = sns.heatmap(Compare)
    plt.savefig('dualheatmapcompare' + str(filename) + '.png')

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

## Package up filtering steps. Just expedient for the moment, revise later
def getfiltereddata(h5_files):
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
    
    return data_auto1_filt,data_auto2_filt


def orientation(data_auto_arg):
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

## Class to handle data manipulation for probabilistic metrics. 

## Analyze the joint distributions of angular metrics. 
class AngularAnalysis(object):
    ## Take in four angle sets as pandas arrays (must be of the same length). The first two represent angles of the fish w.r.t each other, the second two represent opercula angles of the two fish. 
    def __init__(self,angle1,angle2,operangle1,operangle2):
        self.fish1_angle = angle1
        self.fish2_angle = angle2
        self.fish1_operangle = operangle1
        self.fish2_operangle = operangle2
        ## organize for easy indexing:
        self.fish1 = [self.fish1_angle,self.fish1_operangle]
        self.fish2 = [self.fish2_angle,self.fish2_operangle]
        self.fish = [self.fish1,self.fish2]

    ## we have plotting methods and we have probability methods. Within plotting, we have 1d and 2d methods. 2d are for visualization: 

    def plot_2d_face(self,title,timestart = None,timeend = None,kind = 'hex',save = False):
        '''
        Joint desnity of both fish heading direction. 
        title: (string) the title of the figure. 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace. 
        kind: (string) the kind argument passed to seaborn jointplot. 
        save: (bool) whether or not to save the figure 
        '''
        plot = sns.jointplot(self.fish1_angle[timestart:timeend],self.fish2_angle[timestart:timeend],kind = kind)
        
        if save == True: 
            plt.savefig(title+'.pdf')
        plt.show()
        
    def plot_2d_oper(self,title,timestart = None,timeend = None,kind = 'hex',save = False):
        '''
        Joint desnity of both fish heading direction. 
        title: (string) the title of the figure. 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace. 
        kind: (string) the kind argument passed to seaborn jointplot. 
        save: (bool) whether or not to save the figure 
        '''
        plot = sns.jointplot(self.fish1_operangle[timestart:timeend],self.fish2_operangle[timestart:timeend],kind = kind)
        
        plot.ax_joint.axhline(y = 65,color = 'black')
        plot.ax_joint.axvline(x = 65,color = 'black')
        
        if save == True: 
            plt.savefig(title+'.png')
        plt.show()

    def plot_2d_face_shiftfish1(self, title, timestart = None, timeend = None, Shift = 40, kind = 'hex', save = False):
        
        ''' 
        SHIFTS FISH1
        joint density of both fish heading direction with a designated delay time (40 frames for 1 sec in a 40 fps video)
        title: (string) the title of the figure. 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace.
        shift: number of frames to shift dataframe 1 down
        '''
        sns.jointplot(self.fish1_angle.shift(Shift)[timestart:timeend], self.fish2_angle[timestart:timeend], kind = kind)
       
        if save == True: 
            plt.savefig(title+'.png')
        
    def plot_2d_face_shiftfish2(self, title, timestart = None, timeend = None, Shift = 40, kind = 'hex', save = False):
        
        ''' 
        SHIFTS FISH1
        joint density of both fish heading direction with a designated delay time (40 frames for 1 sec in a 40 fps video)
        title: (string) the title of the figure. 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace.
        shift: number of frames to shift dataframe 1 down
        '''
        sns.jointplot(self.fish1_angle[timestart:timeend], self.fish2_angle.shift(Shift)[timestart:timeend], kind = kind)

        if save == True: 
            plt.savefig(title+'.png')
            
    def plot_2d_op_shiftfish1(self, title, timestart = None, timeend = None, Shift = 40, kind = 'hex', save = False):
        
        ''' 
        SHIFTS FISH1
        joint density of both fish heading direction with a designated delay time (40 frames for 1 sec in a 40 fps video)
        title: (string) the title of the figure. 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace.
        shift: number of frames to shift dataframe 1 down
        '''
        plot = sns.jointplot(self.fish1_operangle.shift(Shift)[timestart:timeend], self.fish2_operangle[timestart:timeend], kind = kind)
        
        plot.ax_joint.axhline(y = 69,color = 'black')
        plot.ax_joint.axvline(x = 66,color = 'black')
        
        if save == True: 
            plt.savefig(title+'.png')
        plt.show()
           
    def plot_2d_op_shiftfish2(self, title, timestart = None, timeend = None, Shift = 40, kind = 'hex', save = False):
        
        ''' 
        SHIFTS FISH1
        joint density of both fish heading direction with a designated delay time (40 frames for 1 sec in a 40 fps video)
        title: (string) the title of the figure. 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace.
        shift: number of frames to shift dataframe 1 down
        '''
        plot = sns.jointplot(self.fish1_operangle[timestart:timeend], self.fish2_operangle.shift(Shift)[timestart:timeend], kind = kind)
        
        plot.ax_joint.axhline(y = 69,color = 'black')
        plot.ax_joint.axvline(x = 66,color = 'black')
        
        if save == True: 
            plt.savefig(title+'.png')
        plt.show()
        
    def plot_2d_att(self,title,fishid,timestart = None,timeend = None,kind = 'hex',save = False):
        '''
        Joint density of one fish's heading direction + operculum open width
        title: (string) the title of the figure. 
        fishid: (int) the identiity of the fish to focus on, 0 or 1 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace. 
        kind: (string) the kind argument passed to seaborn jointplot. 
        save: (bool) whether or not to save the figure 
        '''

        fishdata = self.fish[fishid]
        fishangle = fishdata[0]
        fishoper = fishdata[1]
        plot = sns.jointplot(fishangle[timestart:timeend],fishoper[timestart:timeend],kind = kind)
        ## add in black lines on threshold:
        plot.ax_joint.axhline(y = 65,color = 'black')
        plot.ax_joint.axvline(x = 140,color = 'black')
        plot.ax_marg_y.axhline(y = 65,color = 'black')
        plot.ax_marg_x.axvline(x = 140,color = 'black')
        plot.set_axis_labels("Heading Angle", "Operculum Degree");
        if save == True: 
            plt.savefig(title+'.png')
        plt.show()
        
    def plot_2d_att_across(self,title,fishidop,fishidorient, timestart = None,timeend = None,kind = 'hex',save = False):
        '''
        Joint density of one fish's heading direction + operculum open width
        title: (string) the title of the figure. 
        fishid: (int) the identiity of the fish to focus on, 0 or 1 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace. 
        kind: (string) the kind argument passed to seaborn jointplot. 
        save: (bool) whether or not to save the figure 
        '''

        fishop = self.fish[fishidop][1]
        fishorient = self.fish[fishidorient][0]
 
        plot = sns.jointplot(fishorient[timestart:timeend],fishop[timestart:timeend],kind = kind)
        ## add in black lines on threshold:
        plot.ax_joint.axhline(y = 65,color = 'black')
        plot.ax_joint.axvline(x = 140,color = 'black')
        plot.ax_marg_y.axhline(y = 65,color = 'black')
        plot.ax_marg_x.axvline(x = 140,color = 'black')
        plot.set_axis_labels("Heading Angle", "Operculum Degree");
        if save == True: 
            plt.savefig(title+'.pdf')
        plt.show()

    def plot_3d_att_across(self,title,fishidop,fishidorient, timestart = None,timeend = None,kind = 'kde',save = False):
        '''
        Joint density of one fish's heading direction + operculum open width
        title: (string) the title of the figure. 
        fishid: (int) the identiity of the fish to focus on, 0 or 1 
        timestart: (int) the time that we start counting the trace from. 
        timeend: (int) the time that we stop counting the trace. 
        kind: (string) the kind argument passed to seaborn jointplot. 
        save: (bool) whether or not to save the figure 
        '''
        # create dataframe and fill with operculum data for both fish, orientation of one fish
        condition = pd.DataFrame()
        
        condition['fish1op'] = self.fish[fishidop][1][timestart:timeend]
        condition['fish2orient'] = self.fish[fishidorient][0][timestart:timeend]
        condition['fish2op'] = self.fish[fishidorient][1][timestart:timeend]
        
        # change second fish operculum to a binary
#        condition['fish2op'] = condition['fish2op'].apply(lambda x: 1 if x > 65 else 0)
#        
        #sns.lmplot('fish2orient', 'fish1op', data = condition, fit_reg = False, hue = 'fish2op', legend = False, scatter_kws={'alpha':0.05})
        
        condition = condition.dropna()
        
        below = condition[condition['fish2op'] < 65]
        above = condition[condition['fish2op'] >= 65]
        
        belowfish1op = np.array(below['fish1op'])
        belowfish2orient = np.array(below['fish2orient'])
        abovefish1op = np.array(above['fish1op'])
        abovefish2orient = np.array(above['fish2orient'])
        
#        sns.jointplot(condition['fish2orient'], condition['fish1op'], kind = kind)
        ax = sns.kdeplot(belowfish2orient, belowfish1op, cmap="Reds", shade=False, shade_lowest=False)
        ax = sns.kdeplot(abovefish2orient, abovefish1op, cmap="Blues", shade=False, shade_lowest=False)
        

        ax.axhline(y = 65,color = 'black')
#        ax.set_ylim([0,160])
#        ax.set_axis_labels("Heading Angle", "Operculum Degree");
        if save == True: 
            plt.savefig(title+'.pdf')
        plt.show()
        
    ## 1d are conditional distributions: 

    def plot_1d_face(self,title,targetfish,condition = None,timestart = None,timeend = None,save = False):
        '''
        Distribution of one heading angle, optionally conditioned on the values of another being in a certain range. 
        title: (string) the figure title. 
        targetfish: (int) the indentity marker of the fish to focus on. 
        condition: (list) a set of two integers giving the lower and upper limits for an angle in the non-target fish. 
        timestart: (int)
        timeend: (int)
        save: (int)
        '''
        ## First do some data manipulation:
        condfish = abs(1-targetfish)
        targangle = self.fish[targetfish][0]
        condangle = self.fish[condfish][0]
        ## Truncate to a particular range of times: 
        targcrop,condcrop = targangle[timestart:timeend],condangle[timestart:timeend]
        ## Get indices after applying condition in the other fish: 
        condinds = condcrop.index[condcrop.between(*condition)]
        ## Collect relevant data points in target fish: 
        vals = targcrop.loc[condinds]
        ## Do a kde plot on these values
        sns.kdeplot(vals)
        if save == True: 
            plt.savefig(title+'.png')
        plt.show()

    def plot_1d_att(self,title,targetfish,condition = None,timestart = None,timeend = None,save = False):
        '''
        Distribution of one heading angle, optionally conditioned on the values of another being in a certain range. 
        title: (string) the figure title. 
        targetfish: (int) the indentity marker of the fish to focus on. 
        condition: (list) a set of two integers giving the lower and upper limits for an angle in the non-target fish. 
        timestart: (int)
        timeend: (int)
        save: (int)
        '''
        ## First do some data manipulation:
        targangle = self.fish[targetfish][0]
        targoper = self.fish[targetfish][1]
        ## Truncate to a particular range of times: 
        anglecrop,opercrop = targangle[timestart:timeend],targoper[timestart:timeend]
        ## Get indices after applying condition in the other fish: 
        condinds = anglecrop.index[anglecrop.between(*condition)]
        ## Collect relevant data points in target fish: 
        vals = targoper.loc[condinds]
        ## Do a kde plot on these values
        sns.kdeplot(vals)
        if save == True: 
            plt.savefig(title+'.png')
        #plt.show()

    ## 1d are conditional distributions: 
    def prob_1d_face(self,title,targetfish,condition = None,cutoff=180,timestart = None,timeend = None,save = False):
        '''
        Distribution of one heading angle, optionally conditioned on the values of another being in a certain range. 
        title: (string) the figure title. 
        targetfish: (int) the indentity marker of the fish to focus on. 
        condition: (list) a set of two float giving the lower and upper limits for an angle in the non-target fish. 
        cutoff: float() an 
        timestart: (int)
        timeend: (int)
        save: (int)
        '''
        ## First do some data manipulation:
        condfish = abs(1-targetfish)
        targangle = self.fish[targetfish][0]
        condangle = self.fish[condfish][0]
        ## Truncate to a particular range of times: 
        targcrop,condcrop = targangle[timestart:timeend],condangle[timestart:timeend]
        ## Get indices after applying condition in the other fish: 
        condinds = condcrop.index[condcrop.between(*condition)]
        ## Collect relevant data points in target fish: 
        vals = targcrop.loc[condinds]
        ## Calculate the proportion of the data below a cutoff value: 
        prob = len(vals.index[vals>cutoff])/len(vals)
        return prob

    def prob_1d_att(self,title,targetfish,condition = None,cutoff=180,timestart = None,timeend = None,save = False):
        '''
        Distribution of one heading angle, optionally conditioned on the values of another being in a certain range. 
        title: (string) the figure title. 
        targetfish: (int) the indentity marker of the fish to focus on. 
        condition: (list) a set of two float giving the lower and upper limits for an angle in the non-target fish. 
        cutoff: float() an 
        timestart: (int)
        timeend: (int)
        save: (int)
        '''
        ## First do some data manipulation:
        targangle = self.fish[targetfish][0]
        targoper = self.fish[targetfish][1]
        ## Truncate to a particular range of times: 
        anglecrop,opercrop = targangle[timestart:timeend],targoper[timestart:timeend]
        ## Get indices after applying condition in the other fish: 
        condinds = anglecrop.index[anglecrop.between(*condition)]
        ## Collect relevant data points in target fish: 
        vals = targoper.loc[condinds]
        ## Calculate the proportion of the data below a cutoff value: 
        prob = len(vals.index[vals>cutoff])/len(vals)
        return prob

        #print('building histograms...')
        #self.hist_comp,_,_ = np.histogram2d(self.fish1_angle,self.fish2_angle,bins = np.arange(180),density = True)
        #self.hist_1,_,_ = np.histogram2d(self.fish1_angle,self.fish1_operangle,bins = np.arange(180),density = True)
        #self.hist_2,_,_ = np.histogram2d(self.fish2_angle,self.fish2_operangle,bins = np.arange(180),density = True)
        #print('histograms constructed.')

    


if __name__ == "__main__":
    # ##### Load Data

    # In[4]:

    home_dir = '.'#'/Users/Claire/Desktop/Test'
    h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
    print(h5_files)

    ## Packaged up some of the upload code. 
    data_auto1_filt,data_auto2_filt = getfiltereddata(h5_files)

#    excel_files = glob(os.path.join(home_dir, '*.xlsx'))
#    
    ## Groundtruth data
#    file_handle3 = excel_files[0]
#    data_manual1 = pd.read_excel(file_handle3)
#   
#    file_handle4 = excel_files[1]
#    data_manual2 = pd.read_excel(file_handle4)
    
    ## Take the filtered tracked points, and return orientation, opercula angles. 
    angle1 = gaze_tracking(data_auto1_filt,data_auto2_filt)
    angle2 = gaze_tracking(data_auto2_filt,data_auto1_filt)
    Operangle1 = auto_scoring_get_opdeg(data_auto1_filt)
    Operangle2 = auto_scoring_get_opdeg(data_auto2_filt)
    
#    n = 47999
#    m = 71986

    n = 100000
    m = 101000
    c = 144000
    d = 145000
    
    RateFish1_beginning = Percent_Trial(auto_scoring_TS1, data_auto1_filt[a:b])
    RateFish2_beginning = Percent_Trial(auto_scoring_TS1, data_auto2_filt[a:b])
    
    RateFish1_end = Percent_Trial(auto_scoring_TS1, data_auto1_filt[c:d])
    RateFish2_end = Percent_Trial(auto_scoring_TS1, data_auto2_filt[c:d])
    print(RateFish1_beginning, RateFish2_beginning, RateFish1_end, RateFish2_end)
    
    SpeedFish1 = speed(data_auto1_filt[n:m])
    SpeedFish2 = speed(data_auto2_filt[n:m])
    plt.plot(SpeedFish2)
    
    
    def myplot(x, y, s, bins=500):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent
___


#    data_auto1_before = data_auto1_filt[n:m].dropna()
#    data_auto2_before = data_auto2_filt[n:m].dropna()
#    
#    
#    img, extent = myplot(data_auto1_before['A_head']['x'], data_auto1_before['A_head']['y'], 2)
#    fig, ax = plt.subplots()
#    ax.imshow(img, extent=extent, origin='lower', cmap=cm.rainbow, vmin=0, vmax=1)
#    ax.set(xticks=[], yticks=[])
#
#
#    img, extent = myplot(data_auto2_before['A_head']['x'], data_auto2_before['A_head']['y'], 2)
#    fig, ax = plt.subplots()
#    ax.imshow(img, extent=extent, origin='lower', cmap=cm.rainbow, vmin=0, vmax=1)
#    ax.set(xticks=[], yticks=[])
#    crossnum1, crossind1 = findcrossingunder(Operangle2, 69)
#    crossind1barrier = [i for i in crossind1 if i >= n]
#    crossind1barrier = [i for i in crossind1barrier if i <= m]
#    
#    Trunc = np.random.rand(0, 401)
#
#    for i in crossind1barrier:
#        if i > 201:
#            Chunk = Operangle1.truncate(before = i - 200, after = i + 200).T
#            Trunc = np.vstack([Trunc, Chunk])
#        
#    Trunc2 = pd.DataFrame(Trunc.T)
#
#    indbins = np.random.rand(0,len(Trunc2.values))
###
#    for i in np.arange(len(crossind1barrier)):
#        indbin = uniform(binarizeOp(Trunc2[i], threshold = 70), len(Trunc2.values))
#        indbins = np.vstack([indbins, indbin])
#
#    fix, ax = plt.subplots(2,1, sharex = True)
#    ax[0].eventplot(indbins)
##ax[1].hist(indbins[np.where(indbins != 0)], bins = 100)
##plt.savefig('histplusraster.pdf')
#    ax[1].set_xlim([0,len(Trunc2.values)])
#    A = sns.distplot(indbins[np.where(indbins != 0)], bins = 100, ax = ax[1])
##
#    plt.savefig('VM1offsetVM3readout.pdf') 
#    A = AngularAnalysis(angle1, angle2, Operangle1, Operangle2)
#    
##    A.plot_2d_att_across("attFish2op_across", 1, 0, n, n + 10000, kind = "kde", save = True)
##    A.plot_2d_att_across("attFish2op_across", 0, 1, n, n + 10000, kind = "kde", save = True)
#    A.plot_3d_att_across('try', 0, 1, timestart = n, timeend= n + 10000, kind = 'kde', save = True)
### making the original kdeplots/1D plots

    
#87525:154600
#92074:164012
#73544, 144038
#    314vs315
#    n = 73261
#    m = 144039
#    VM4vsVM5
#n = 88119
#m = end
#    VM3 VM4
#        n = 3166
#    m = 71524
#    VM1 VM3
#        n = 70775
#    m = 138051
    
    ## Makes the 1D kdeplots for all fish, overlap in same graph
#    B = A.plot_1d_att("IF1_IF4Fish1", 1,[140,180],n, m, True)
#    B = A.plot_1d_att("IF1_IF4Fish1", 1,[0,140],n, m, True)
##    
#    
#    C = A.plot_1d_att("IF1_IF4Fish2", 0,[140,180],n, m, True)
#    C = A.plot_1d_att("IF1_IF4Fish2", 0,[0,140],n, m, True)
 
    
#    
#    ## Makes 2D facing kdeplot and the att plots for each fish at 5 min intervals
#    list1 = [n, n + 10000]
#    counter = 0
#    for i in list1:
#       if counter < len(list1)-1:
#           A.plot_2d_att_across("attFish1op_across_help" + str(counter), 0,1, list1[counter], list1[counter + 1], kind = "kde", save = True)
#          
#           A.plot_2d_att_across("attFish2op_across_help" + str(counter), 1,0, list1[counter], list1[counter + 1], kind = "kde", save = True)
##           
##           ## NOTE: This has is the entirety of code change. 
##           A.plot_2d_face("face" + str(counter), list1[counter], list1[counter + 1], kind = "kde", save = True)
##           A.plot_2d_oper("oper" + str(counter), list1[counter], list1[counter + 1], kind = "kde", save = True)
#           
#           counter = counter + 1
#
#    


## jitter facing 2D - kdeplots
    
    ## Captures different amounts of facing jitter(shifts) within different windows of the trial
#    
#    counter = 0
#    shifter = 0
#    list1 = [n + 10000, m]
#    shifts1 = [-1323, -700, -600, -500, 0, 500, 600, 700, 1323]
#    for i in list1:
#        if counter < len(list1)-1:
#            for i in shifts1:
##                A.plot_2d_face_shiftfish1("Fish1shift" + str(list1[counter]) + "_" + str(shifts1[shifter]), list1[counter], list1[counter +1], shifts1[shifter], kind = "kde", save = True)
#                
#                A.plot_2d_op_shiftfish2("Fish2att" + str(list1[counter]) + "_" + str(shifts1[shifter]), list1[counter], list1[counter +1], shifts1[shifter], kind = "kde", save = True)
#                
#                shifter = shifter + 1
#                
#            counter = counter + 1
#            shifter = shifter - len(shifts1)
##    

         
## jitter operculum - kdeplots
            
## measuring probability of oper when facing
            
## measure probability of oper2 when oper1
    
###                 *** NO NEED TO GO PAST THIS POINT****
    
    
#    
#    counter = 0
#    shifter = 0
#    list1 = [0, 70000, 90000]
#    shifts1 = [0, 20]
#    for i in list1:
#        if counter < len(list1)-1:
#            for i in shifts1:
#                print(counter, shifter)
#                shifter = shifter + 1
#                
#        counter = counter + 1
#        shifter = shifter - len(shifts1)  
#    
    
    
    
    
    
    
###    plt.show()
#   
           
#
#    HeatmapCompare('angle1IM1_IM2', angle1, data_manual1, 88150, 88910)
#    HeatmapCompare('angle2IM1_IM2', angle2, data_manual2, 88150, 88910)
#    DualAngleHeatMapCompare('daulangleIM1_IM2', angle1, angle2, data_manual1, data_manual2, 88150, 88910)
        

    ##Making joint kdeplots to assess dual orientation of continuous angles
#    n = 92074
#    list1 = [n- 10000, n, n + 10000, n + 20000, n + 30000, n + 40000, n + 50000, n + 60000, n + 70000]
#    counter = 0
#    for i in list1:
#        x1 = pd.Series(angle1[list1[counter]:list1[counter + 1]], name="$X_1$")
#        x2 = pd.Series(angle2[list1[counter]:list1[counter + 1]], name="$X_2$")
#        g = sns.jointplot(x1, x2, kind="kde", height=7, space=0)
#        plt.savefig('jointkdeplot' + str(counter) + '.pdf')
#        counter = counter + 1
    
    
    
##        
  

#    data_auto1_filt['zeroed','x'] = data_auto1_filt['A_head']['x'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[0]
#    data_auto1_filt['zeroed','y'] = data_auto1_filt['A_head']['y'] - midpoint(data_auto1_filt['B_rightoperculum']['x'], data_auto1_filt['B_rightoperculum']['y'], data_auto1_filt['E_leftoperculum']['x'], data_auto1_filt['E_leftoperculum']['y'])[1]
#    
#    data_auto2_filt['zeroed','x'] = data_auto2_filt['A_head']['x'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[0]
#    data_auto2_filt['zeroed','y'] = data_auto2_filt['A_head']['y'] - midpoint(data_auto2_filt['B_rightoperculum']['x'], data_auto2_filt['B_rightoperculum']['y'], data_auto2_filt['E_leftoperculum']['x'], data_auto2_filt['E_leftoperculum']['y'])[1]
#    
    
 
    
    
    
#    n = 92074
#    list1 = [n- 10000, n, n + 10000, n + 20000, n + 30000, n + 40000, n + 50000, n + 60000, n + 70000]
#    counter = 0
#    for i in list1:
#        x1 = pd.Series(angle2[list1[counter]:list1[counter + 1]], name="$X_1$")
#        x2 = pd.Series(Operangle2[list1[counter]:list1[counter + 1]], name="$X_2$")
#        g = sns.jointplot(x1, x2, kind="kde", height=7, space=0)
#        plt.savefig('jointkdeplot' + str(counter) + '.pdf')
#        counter = counter + 1
#    
  
    
#    x1 = pd.Series(angle1[92074:164012], name="$X_1$")
#    x2 = pd.Series(Operangle1[92074:164012], name="$X_2$")
#    x3 = pd.Series(angle2[92074:164012], name="$X_1$")
#    x4 = pd.Series(Operangle2[92074:164012], name="$X_2$")
    

    
    # Set up the figure
#    f, ax = plt.subplots(figsize=(8, 8))
#    ax.set_aspect("equal")
#
## Draw the two density plots
#    ax = sns.kdeplot(x1, x2,
#                 cmap="Reds", shade=True, shade_lowest=False)
#    ax = sns.kdeplot(x3, x4,
#                 cmap="Blues", shade=True, shade_lowest=False)
#
#    red = sns.color_palette("Reds")[-2]
#    blue = sns.color_palette("Blues")[-2]
#    ax.text(2.5, 8.2, "Orientation", size=16, color=blue)
#    ax.text(3.8, 4.5, "Operculum", size=16, color=red)
#    
#    if i in angle1 > 180:
#        print "jey"
    
    # conditional, new array, kdeplot of new array
    # integral, of probability 
    
    
    
    #gaze_ethoplot([angle1,angle2],'test',show = True, save = False)
    #print (binarize([angle1,angle2]))
    
