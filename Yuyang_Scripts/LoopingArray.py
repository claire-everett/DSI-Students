#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:18:36 2020

@author: ryan
"""
import numpy as np
class LoopingArray():
    #1D looping array
    def __init__(self,array):
        self.data = array

    def __getitem__(self, key):
        l=len(self.data)
        try:
            #key is an integer
            return self.data[key%l]
        except:
            #key is a slice
            start=key.start
            stop=key.stop
            step=key.step
            if start is None:
                start=0
            if stop is None:
                stop=l
            elif stop<0:
                stop=l-stop+1
            if step is None:
                step=1
            start=start%l
            stop=(stop-1)%l+1
            while stop<start:
                stop=stop+l
            re=[]
            for i in range(start,stop,abs(step)):
                re.append(self.data[i%l])
            re=np.array(re)
            if step<0:
                re=re[::-1]
            return np.array(re)
    def __setitem__(self, key, value):
        l=len(self.data)
        try:
            self.data[key%l] = value
        except:
            ValueError("set item other than using integer not implemented yet")

    def __repr__(self):
        return "LoopingArray({})".format(self.data)

    def __len__(self):
        return len(self.data)

