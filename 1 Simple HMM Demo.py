
# coding: utf-8

# # Start with a simple Hidden Markov Model

# In[29]:
from glob import glob
import seaborn as sns; sns.set()
import pandas as pd
import os
import autograd.numpy as np
import autograd.numpy.random as npr
import pandas as pd
npr.seed(0)


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

def auto_scoring_TS1(data_auto,thresh_param0 = 65.45,thresh_param1 = 135.56):
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
     
    return data_auto1_filt



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

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


# In[2]:


# Set the parameters of the HMM
T = 13069   # number of time bins
K = 2       # number of discrete states
D = 2       # data dimension

# Make an HMM
true_hmm = ssm.HMM(K, D, observations="gaussian")

# Manually tweak the means to make them farther apart
thetas = np.linspace(0, 2 * np.pi, K, endpoint=False)
true_hmm.observations.mus = 3 * np.column_stack((np.cos(thetas), np.sin(thetas)))


# In[3]:


# Sample some data from the HMM
z, y = true_hmm.sample(T)
true_ll = true_hmm.log_probability(y)

home_dir = '.'#'/Users/Claire/Desktop/Test'
h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
print(h5_files)

## Packaged up some of the upload code. 
data_auto1_filt = getfiltereddata(h5_files)

#data = auto_scoring_get_opdeg(data_auto1_filt)[88119:len(data_auto1_filt)]
XX = data_auto1_filt["A_head"]["x"]
YY = data_auto1_filt["A_head"]["y"]

data1 = np.column_stack((XX, YY))
data = data1.dropna(data)
# In[4]:


# Plot the observation distributions
#lim = .85 * abs(y).max()
#XX, YY = np.meshgrid(np.linspace(-lim, lim, 100), np.linspace(-lim, lim, 100))
#data = np.column_stack((XX.ravel(), YY.ravel()))


input = np.zeros((data.shape[0], 0))
mask = np.ones_like(data, dtype=bool)
tag = None
lls = true_hmm.observations.log_likelihoods(data, input, mask, tag)


# In[36]:


plt.figure(figsize=(6, 6))
for k in range(K):
    plt.contour(XX, YY, np.exp(lls[:,k]).reshape(XX.shape), cmap=white_to_color_cmap(colors[k]))
    plt.plot(y[z==k, 0], y[z==k, 1], 'o', mfc=colors[k], mec='none', ms=4)
    
plt.plot(y[:,0], y[:,1], '-k', lw=1, alpha=.25)
plt.xlabel("$y_1$")
plt.ylabel("$y_2$")
plt.title("Observation Distributions")

if save_figures:
    plt.savefig("hmm_1.pdf")


# In[35]:


# Plot the data and the smoothed data
lim = 1.05 * abs(y).max()
plt.figure(figsize=(8, 6))
plt.imshow(z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1, extent=(0, T, -lim, (D)*lim))

Ey = true_hmm.observations.mus[z]
for d in range(D):
    plt.plot(y[:,d] + lim * d, '-k')
    plt.plot(Ey[:,d] + lim * d, ':k')

plt.xlim(0, T)
plt.xlabel("time")
plt.yticks(lim * np.arange(D), ["$y_{}$".format(d+1) for d in range(D)])

plt.title("Simulated data from an HMM")

plt.tight_layout()

if save_figures:
    plt.savefig("hmm_2.pdf")


# # Fit an HMM to this synthetic data

# In[7]:


N_iters = 50
hmm = ssm.HMM(K, D, observations="gaussian")
hmm_lls = hmm.fit(y, method="em", num_em_iters=N_iters)

plt.plot(hmm_lls, label="EM")
plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")


# In[8]:


# Find a permutation of the states that best matches the true and inferred states
hmm.permute(find_permutation(z, hmm.most_likely_states(y)))


# In[11]:


# Plot the true and inferred discrete states
hmm_z = hmm.most_likely_states(y)

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, T)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, T)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()


# In[34]:


# Use the HMM to "smooth" the data
hmm_y = hmm.smooth(y)

plt.figure(figsize=(8, 4))
plt.plot(y + 3 * np.arange(D), '-k', lw=2)
plt.plot(hmm_y + 3 * np.arange(D), '-', lw=2)
plt.xlim(0, T)
plt.ylabel("$y$")
# plt.yticks([])
plt.xlabel("time")

