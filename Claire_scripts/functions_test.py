#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:50:16 2020

@author: Claire
"""

from hmmlearn import hmm
import numpy as np
import pandas as pd
from glob import glob
import os

def uniform(indbin,LoopLength):
    '''
    Takes indbin
    '''
    A = len(indbin)
    B = LoopLength
    C = B - A
    end = np.zeros(C)
    new = np.concatenate((indbin, end))
    
    return(new)

    
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
    
    print(Manual[data_manual['Start'][i]:data_manual['Stop'][i]]) 
     
    return Manual['OpOpen'][crop0:crop1]

def binarizeOp(Operangle, threshold):
    

    boolean = Operangle.apply(lambda x: 1 if x > threshold else 0).values
#    boolean = binary_dilation(boolean, structure = np.ones(40,))
    
    binindex = np.where(boolean)[0]

    return (binindex)


def binarize_Op_2(Operangle, threshold = 72):
    boolean = Operangle.apply(lambda x: 1 if x > threshold else 0).values
    return(boolean)

