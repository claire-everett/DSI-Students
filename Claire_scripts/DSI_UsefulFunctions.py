#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:29:30 2020

@author: Claire
"""

## Loading the Data

# home_dir = '.'#'/Users/Claire/Desktop/Test'
# h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
# print(h5_files)

# ## Packaged up some of the upload code. 
# data_auto1_filt,data_auto2_filt = getfiltereddata(h5_files)

#Usefull functions, please read through
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from functions_test import binarize_Op_2

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
    AvgSpeed = Speed.mean()
#    
    return AvgSpeed


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
    data_manual['Stop']
    for i in reference:
        Manual[data_manual['Start'][i]:data_manual['Stop'][i]] = 1
    return Manual['OpOpen'][crop0:crop1]


## Now we will start writing functions to output the result of analyzing the behavioral traces to give automatic scoring: 
def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    return (data['x'],data['y'])

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

    return lawofcosines(HROP,HLOP,RLOP)

## Additional parameters for smoothing could be taken
def auto_scoring_smooth_opdeg(opdeg):
    #Smoothes OPdeg
    x=range(0,len(opdeg),1)
    w = np.isnan(opdeg)
    opdeg[w] = 0.
    u_p=UnivariateSpline(x,opdeg,w = ~w)
#     d = {'xcoord': range(0,len(opdeg),1), 'raw': opdeg,'smoothed':u_p}
    return pd.Series(u_p)

## Filters by width of detected events. 
def auto_scoring_widthfilter(binary_scores,widththresh = 30):

    fst = binary_scores.index[binary_scores & ~ binary_scores.shift(1).fillna(False).astype(bool)]
    lst = binary_scores.index[binary_scores & ~ binary_scores.shift(-1).fillna(False).astype(bool)]
    
    print(len(fst))
    if len(fst) < 1:
        width_filtered = pd.DataFrame(0, index=np.arange(len(binary_scores)), columns = ['OpOpen'])
    
    if len(fst) > 0:
        intv = pd.DataFrame([(i, j) for i, j in zip(fst, lst) if j > i + 10]) ## 10 is also a parameter..
        intv.columns=['start', 'end']
        intv['width'] = intv['end']-intv['start']
        intv = intv.loc[intv['width'] > widththresh]
        intv['new_col'] = range(0, len(intv))

        reference = intv.index
        width_filtered = pd.DataFrame(0, index=np.arange(len(binary_scores)), columns = ['OpOpen'])
        for i in reference:
            width_filtered[intv['start'][i]:intv['end'][i]] = 1

        return width_filtered

def auto_scoring_tracefilter(data,p0=20,p1=250,p2=15,p3=70,p4=200):
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum', 'C_tailbase', 'D_tailtip','E_leftoperculum']
    mydata['bodylength'] = mydistance(coords(mydata[boi[0]]),coords(mydata[boi[3]]))
    mydata['Operwidth'] = mydistance(coords(mydata[boi[4]]),coords(mydata[boi[1]]))
    mydata['HeadROperwidth'] = mydistance(coords(mydata[boi[0]]),coords(mydata[boi[1]]))
    mydata['HeadLOperwidth'] = mydistance(coords(mydata[boi[0]]),coords(mydata[boi[4]]))
    mydata['TailtipROperwidth'] = mydistance(coords(mydata[boi[3]]),coords(mydata[boi[1]]))
    mydata['TailtipLOperwidth'] = mydistance(coords(mydata[boi[3]]),coords(mydata[boi[4]]))
    mydata['TailbaseLOperwidth'] = mydistance(coords(mydata[boi[2]]),coords(mydata[boi[4]]))
    mydata['TailbaseROperwidth'] = mydistance(coords(mydata[boi[2]]),coords(mydata[boi[1]]))

    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
    #         print (xdiff_check.loc[xdiff_check == True])
            mydata[xdiff_check] = np.nan
    #         print (mydata.loc[np.isnan(mydata['A_head']['x'])])

            bodylength_check = mydata['bodylength'] > p1
            mydata[bodylength_check] = np.nan

            origin_check = mydata[b][j] < p2
            mydata[origin_check] = np.nan

            Operwidth_check = mydata['Operwidth'] > p3
            mydata[Operwidth_check] = np.nan

            HeadROperwidth_check = mydata['HeadROperwidth'] > p3
            mydata[HeadROperwidth_check] = np.nan

            HeadLOperwidth_check = mydata['HeadLOperwidth'] > p3
            mydata[HeadLOperwidth_check] = np.nan

            TTL_check = mydata['TailtipLOperwidth'] > p4
            mydata[TTL_check] = np.nan

            TTR_check = mydata['TailtipROperwidth'] > p4
            mydata[TTR_check] = np.nan
    return mydata

##############################################################################
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


## Take the filtered tracked points, and return orientation, opercula angles. 
