#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:56:09 2020

@author: ryan
"""

import os
os.chdir("/Users/ryan/Desktop/Fish Project")
import functions 
from functions import *
from glob import glob
import numpy as np
from hmmlearn import hmm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
def sin_curve(x, b,c,d):
    return  30*np.sin(b * x+c)+d
def find_params(periculum,start=100000,end=100400):
    params, params_covariance = optimize.curve_fit(sin_curve, range(start,end), periculum[start:end],p0=[0.05,0,70])
    return params

def find_error(periculum):
    params=find_params(periculum)
    predict=sin_curve(range(len(periculum)),params[0],params[1],params[2])
    error=predict-periculum
    return error