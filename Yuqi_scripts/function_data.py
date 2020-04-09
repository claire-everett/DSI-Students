#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:56:08 2020

@author: miaoyuqi
"""

import ssm
import h5py
import pandas as pd
f = h5py.File('/Users/miaoyuqi/研究/Statistical analyses of Siamese fighting fish aggressive behavior/DSI-Students/Yuqi_scripts/top example.h5', 'r')
top = f["df_with_missing"]
pd.DataFrame(f)
