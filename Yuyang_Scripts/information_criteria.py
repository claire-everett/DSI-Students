#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:15:19 2020

@author: ryan
"""
import numpy as np
import math
def num_params(model,dim):
    n=model.n_components
    return n-1+n*(n-1)+n*dim+n*(dim*dim+dim)/2

def BIC(model,obs,dim):
    return -2*model.score(obs)+num_params(model,dim)*math.log(obs.shape[0])

def AIC(model,obs,dim):
    return -2*model.score(obs)+2*num_params(model,dim)