#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:13:23 2019

@author: Claire
"""
import pickle
from glob import glob
import seaborn as sns
import pandas as pd
import os
import autograd.numpy as np
import autograd.numpy.random as npr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.colors import LinearSegmentedColormap
import ssm
from ssm.util import find_permutation

def white_to_color_cmap(color, nsteps=256):
    # Get a red-white-black cmap
    cdict = {'red': ((0.0, 1.0, 1.0),
                       (1.0, color[0], color[0])),
                'green': ((0.0, 1.0, 1.0),
                          (1.0, color[1], color[0])),
                'blue': ((0.0, 1.0, 1.0),
                         (1.0, color[2], color[0]))}
    cmap = LinearSegmentedColormap('white_color_colormap', cdict, nsteps)
    return cmap

def gradient_cmap(colors, nsteps=256, bounds=None):
    # Make a colormap that interpolates between a set of colors
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = np.linspace(0,1,ncolors)


    reds = []
    greens = []
    blues = []
    alphas = []
    for b,c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues),
             'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap



sns.set_style("white")
sns.set_context("talk")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

import ssm
from ssm.util import find_permutation

# Speficy whether or not to save figures
save_figures = True


hmm = pickle.load(open('HmmPreserve', 'rb'))


#%%

## Makes Transition Plot
matplotlib.rc_file_defaults()

means = pd.DataFrame(hmm.observations.mus)


F = hmm.transitions.transition_matrix

B = F[~np.eye(F.shape[0],dtype=bool)].reshape(F.shape[0],-1)
B = np.copy(F)
B[np.where(np.eye(B.shape[0]))] = np.nan

im = plt.imshow(B, cmap='gray')

plt.savefig("transitionmatrix.pdf")

#%%

## Emission Plot plus Variance 

sigmas = hmm.observations.Sigmas

variance = []
for i in np.arange(4):
    state = sigmas[i]
    var = np.diag(state)
    variance.append(var)
StateSD = np.sqrt(np.stack(variance,axis = 0)).T
fig,ax = plt.subplots()
colors = ["cyan","purple","orange"]
coords = ["operculum","HeadX","HeadY"]
for mi,m in enumerate(means.values.T):
    print(m)
    sd = StateSD[mi]
    ax.plot(m,color = colors[mi],label = coords[mi])
    ax.plot(m-sd,color = colors[mi],linestyle = '--')
    ax.plot(m+sd,color = colors[mi],linestyle = '--')

#plt.plot(means[means.columns[:3]], "b")
#plt.plot(means[means.columns[:3]]-StateSD,"--")
#plt.plot(means[means.columns[:3]]+StateSD,"--")
ax.set_xticklabels(['blue', 'red', 'yellow', 'green'])
ax.set_xticks(range(4))
plt.legend(loc = 0)

plt.savefig("statemeansSD.pdf")

#%%
# Plot the true and inferred discrete states
hmm_z = hmm.most_likely_states(ytrue)

#plt.figure(figsize=(8, 4))
#plt.subplot(211)
#plt.imshow(z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
#plt.xlim(0, T)
#plt.ylabel("$z_{\\mathrm{true}}$")
#plt.yticks([])

plt.subplot(212)
plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, T)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.xlabel("time")

plt.tight_layout()


#%%