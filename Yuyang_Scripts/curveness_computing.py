#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:16:15 2020

@author: ryan
"""


import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
from moviepy.video.fx.all import crop
from moviepy.editor import VideoFileClip, VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
from tqdm import tqdm
from scipy.ndimage import zoom  
import warnings
#get the images,contours and masks
image_length=600
fps=40
def filter_lowerright(image):
    #this is just a function to help ignore the sign "888" in the lower right corner
    image[500:,:]=255
    return image
#get the contours
def find_contour(videopath,length=40*300,iterpolate=True,step=1):
    vidcap = cv2.VideoCapture(videopath)
    img_array=[]
    mask_array=[]
    contour_array=[]
    length=int(length/step)
    index=0
    for i in tqdm(range(length)):
        success,image=vidcap.read()
        index+=step
        vidcap.set(1,index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        th = cv2.dilate(th, k2, iterations=1)
    #th=cv2.erode(th,None,iterations=1)
        th=filter_lowerright(th)
        contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        flag=0
        area=0
        for cnt in contours:
            new_area=cv2.contourArea(cnt)
            if new_area>area and new_area<10000 :#in case some contour contains almost the whole image, not required if invert the image first
                area=new_area
                fish_contour=cnt
                flag=1
                if flag==0:
                    img=np.float32(np.full(image.shape,255))#just in case there's no valid contour, won't happen in the current case
                else:
                    img=cv2.drawContours(np.float32(np.full(image.shape,255)),[fish_contour],0,(0,0,0),cv2.FILLED)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=np.invert(np.array(cv2.erode(img,k2,iterations=1),np.uint8))
        if iterpolate==True:
            img=zoom(img,3)
            img=cv2.GaussianBlur(img, (9, 9), 0) #since zoom made the img larger
        else:
             img=cv2.GaussianBlur(img, (3, 3), 0)
        true_contour, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_array.append(true_contour[0])
        img_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1],3),255)),true_contour,0,(0,0,0),2),np.uint8))
        mask_array.append(np.array(cv2.drawContours(np.float32(np.full((img.shape[0],img.shape[1],3),255)),true_contour,0,(0,0,0),cv2.FILLED),np.uint8))
    return img_array,mask_array,contour_array
    
img_array,mask_array,contour_array=find_contour("videos/Reverse_888_cut.mp4",length=40*30,step=1)

#save result
'''
out = cv2.VideoWriter('videos/reverse888_contour_blurred.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(img_array)):
     out.write(img_array[i])
out.release()

out = cv2.VideoWriter('videos/reverse888_mask_blurred.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_length,image_length))
for i in range(len(mask_array)):
     out.write(mask_array[i])
out.release()
'''
#%%
#img_array is now point set of masks
#I try to filter the deeplabcut result based on whether they are inside the contour(dilated), but in the end I find 
#another way to calculate curvatue so it's not used at all
'''
path = "Yuyang_contour/Reverse_888DeepCut_resnet50_DLC_toptrackFeb27shuffle1_160000.h5"
f = pd.HDFStore(path,'r')
df = f.get('df_with_missing')
df.columns = df.columns.droplevel()
#the period I used in contour finding
df=df[12000:24000]


colors=["rosybrown","lightcoral","bisque","burlywood","darkorange","darkgoldenrod","gold","olive","olivedrab","forestgreen",
        "lightseagreen","darkslategrey","cyan","dodgerblue"]
labels=["head","Roper","tailbase","tailtip","Loper","spine1","spine2","spine3","spine4","spine5","spine6","spine7",
        "Leye","Reye"]
for i in range(10):
    img=cv2.cvtColor(img_array[i], cv2.COLOR_BGR2GRAY)
    eroded_img=cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (11,11)), iterations=1)
    plt.figure()
    plt.imshow(eroded_img,"gray")
    for j in range(14):
        plt.plot(df.iloc[i,j*3],df.iloc[i,j*3+1],"ro",markersize=2, color=colors[j],label=labels[j])
    plt.legend(loc="upper left",prop={'size': 10},bbox_to_anchor=(1.1, 1.05)) #should be better ways to set legends
    plt.show()

#a video about how original points looks like on contour
fig, ax = plt.subplots(dpi=150)
def make_frame(t):
    t=np.int(t*40)
    ax.clear()
    ax.imshow(img_array[t])
    for j in range(14):
        ax.plot(df.iloc[t,j*3],df.iloc[t,j*3+1],"ro",markersize=2, color=colors[j],label=labels[j])
    ax.legend(loc="upper left",prop={'size': 10},bbox_to_anchor=(1.1, 1.05)) #should be better ways to set legends
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = 300)

#filtering based on mask
def inside_mask(df,mask_array,kernel_size=11):
    data = df.copy()
    for i in tqdm(range(df.shape[0])):
        for j in range(int(df.shape[1]/3)):#how many features in the data
            x=int(np.round(df.iloc[i,j*3]))
            y=int(np.round(df.iloc[i,j*3+1]))
            mask=cv2.cvtColor(mask_array[i], cv2.COLOR_BGR2GRAY)
            k=cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
            eroded_mask=cv2.erode(mask, k, iterations=1)
            if eroded_mask[y,x]!=0:
                data.iloc[i,j*3]=np.nan
                data.iloc[i,j*3+1]=np.nan
    return data


filtered_df=inside_mask(df,mask_array)


#after_filtering comparision
def make_frame(t):
    t=np.int(t*40)
    ax.clear()
    ax.imshow(img_array[t])
    for j in range(14):
        ax.plot(filtered_df.iloc[t,j*3],filtered_df.iloc[t,j*3+1],"ro",markersize=2, color=colors[j],label=labels[j])
    ax.legend(loc="upper left",prop={'size': 10},bbox_to_anchor=(1.1, 1.05)) #should be better ways to set legends
    return mplfig_to_npimage(fig)

animation2 = VideoClip(make_frame, duration = 300)
final_clip = clips_array([[animation, animation2]]).subclip(0,60)
final_clip.write_videofile("videos/contour_bodyparts_compare.mp4",fps=40)

new_df=filtered_df.fillna(method="ffill").fillna(0)
'''

#%%
#calculate the curvature at each point of the contour,default step=3
'''
#currently only testing cosine angle
def compute_pointness(contour, n=3):
    out=np.zeros((600,600))
    contour=contour.squeeze()
    N = len(contour)
    t=1/N
    for i in range(N):
        x_cur, y_cur = contour[i]
        x_next, y_next = contour[(i + n) % N]
        x_prev, y_prev = contour[(i - n) % N]
        dy1=(y_next-y_prev)/(2*n*t)
        dx1=(x_next-x_prev)/(2*n*t)
        dy2=(y_next+y_prev-2*y_cur)/(n*n*t*t)
        dx2=(x_next+x_prev-2*x_cur)/(n*n*t*t)
        curvature=(dx1*dy2-dy1*dx2)/(np.power(dx1*dx1+dy1*dy1,3/2))
        curvature=abs(curvature)
        out[x_cur,y_cur]=curvature
    return out


#test the curvature result, and it seems to make sense as the head or tail has the highest curvature because it's probably the sharpest,
#actions need to ignore that to get valid "curveness" in fish
contour=contour_array[0]
plt.figure()
for i in contour:
    plt.plot(i.flatten()[0],i.flatten()[1],"ro",markersize=1)

out=compute_pointness(contour, n=3)
'''
def find_centroid(contour):
    moments=cv2.moments(contour)
    m01=moments.get("m01")
    m10=moments.get("m10")
    m00=moments.get("m00")
    xbar=m10/m00
    ybar=m01/m00
    return xbar,ybar

def compute_dist(contour,xbar,ybar):
    contour=contour.squeeze()
    dist=np.linalg.norm(contour-np.array([xbar,ybar]),axis=1)
    return dist



def plot_result(curvatures,contour,img_size=600,quantile=0.5):
    head_tail_x,head_tail_y=np.where(curvatures<0)
    new_curvatures=curvatures.copy()
    new_curvatures[new_curvatures<0]=0
    x,y=np.nonzero(new_curvatures)
    xbar,ybar=find_centroid(contour)
    colors=np.array(np.zeros_like(x),np.float64)
    dists=compute_dist(contour,xbar,ybar)
    for i in range(len(x)):
        colors[i]=new_curvatures[x[i],y[i]]
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.scatter(head_tail_x,head_tail_y,s=3,c="lemonchiffon")
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.2)#
    plt.plot(xbar,ybar,"ro",markersize=3,color="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.title("curveness heatmap on fish contour")
    plt.show()
'''
def filter_curvature(curvatures,contour):
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    quantile=np.quantile(dists,0.6)
    contour=contour.squeeze()
    N=len(contour)
    
    for i in range(N):
        x,y=contour[i]
        if dists[i]>quantile:
            curvatures[x,y]=0.1
    return curvatures

#test it again on the first contour
contour=contour_array[0]
out=compute_pointness(contour)
plot_result(out,contour)
filtered_output=filter_curvature(out,contour)
plot_result(filtered_output,contour)


#test on randomly selected 10 contours
samples=np.random.randint(0,12000,10)

for i in samples:
    contour=contour_array[i]
    out=compute_pointness(contour)
    filtered_output=filter_curvature(out,contour)
    plot_result(out,contour)


#compute the curvatures, maybe some tests on how valid the curvature later
#the bad thing is, it's likely the curvature is not accurately measures how fish curves its body, sometimes there might 
#just be a tweek in the tail or fin which creates a sharp area
#possible improvements:
#increase sample steps(n in compute pointness)
#gaussian blur?
#use cos angle instead
#filter out head and tail before hand?(i.e. it x+i/x-i reaches head/tail limit like quantile 0.9 or something, use that latest point)
#average the result with different step length? increase consistency
#edge???


from datetime import datetime
t1=datetime.now()
max_curvature=[]
for contour in tqdm(contour_array):
    curvatures=compute_pointness(contour)
    filtered_curvature=filter_curvature(curvatures,contour)
    max_curvature.append(np.max(filtered_curvature))
t2=datetime.now()
#1 min 17s for 12000 pts
print(t2-t1)

plt.plot(max_curvature)
plt.ylim(0,1)
'''
#find the flip point when going in/out from head/tail
#need to revise it a little bit for dropping small segmentations
def find_anchor(contour,validity=None,cooldown=30):
    #the 2 valid segment here should be in1-out1, in2-out2
    #return the index of the intersection point
    in1=-1
    out1=-1
    in2=-1
    out2=-1
    contour=contour.squeeze()
    #if the valid pts are given before hand, skip this step
    if validity is None:
        xbar,ybar=find_centroid(contour)
        dists=compute_dist(contour,xbar,ybar)
        quantile=np.quantile(dists,0.5)
        validity=dists<quantile
    N = len(contour)
    start=validity[0]
    flag=validity[0]
    warning_cnt=0
    counter=cooldown+1
    if start==0:
        for i in range(N):
            #monitor the point where validity flips
            if flag==0 and validity[i]==1 and counter>cooldown:
                if in1==-1:
                    in1=i
                elif in2==-1:
                    in2=i
                else:
                    warning_cnt+=1
                    print("more than 2 segments detected, already happens {} times".format(warning_cnt))
                counter=0
                flag=1
            elif flag==1 and validity[i]==0 and counter>cooldown:
                if out1==-1:
                    out1=i
                else:
                    out2=i
                flag=0
                counter=0
            else:
                counter+=1
                
    if start==1:
        for i in range(N):
            #monitor the point where validity flips
            if flag==0 and validity[i]==1 and counter>cooldown:
                if in1==-1:
                    in1=i
                elif in2==-1:
                    in2=i
                else:
                    warning_cnt+=1
                    print("more than 2 segments detected, already happens {} times".format(warning_cnt))
                flag=1
                counter=0
            elif flag==1 and validity[i]==0:
                if out2==-1:
                    out2=i
                else:
                    out1=i
                flag=0   
                counter=0
            else:
                counter+=1
    return in1,out1,in2,out2


def compute_cos(contour, step=3,img_size=600,min_step=2):
    #I plan to return 3 objects an img_sizeximg_size array with each point having the cosine angle value on it
    # 2 lists which is the cosine in left and right part
    out=np.zeros((img_size,img_size))
    left_cosines=[]
    right_cosines=[]
    contour=contour.squeeze()
    xbar,ybar=find_centroid(contour)
    dists=compute_dist(contour,xbar,ybar)
    quantile=np.quantile(dists,0.5)
    validity=dists<quantile
    in1,out1,in2,out2=find_anchor(contour,validity)
    N = len(contour)
    t=1/N#actually no use cause the denomiator here cancelled out in later calculation
    def find_next(i,step,min_step):
        #find next point avaliable
        #say if index i+step is in the valid pts, then use i+step as tbe next point
        #otherwise backtrack the points until the point is valid or meet the minimal length requirement
        if step<=min_step or validity[(i+step)%N]==True:
            return contour[(i+step) % N]
        else:
            return find_next(i,step-1,min_step)
    def find_prev(i,step,min_step):
        if step<=min_step or validity[(i-step)%N]==True:
            return contour[(i -step) % N]
        else:
            return find_prev(i,step-1,min_step)
    for i in range(N):
        x_cur,y_cur=contour[i]
    for i in range(N):
        x_cur, y_cur = contour[i]
        if validity[i]==False:
            #head/tail position's cosine angle encoded to -1 for future visualization
            out[x_cur,y_cur]=-1
        else:
            x_next, y_next = find_next(i,step,min_step)
            x_prev, y_prev = find_prev(i,step,min_step)
            vec1=np.array([x_next-x_cur,y_next-y_cur])
            vec2=np.array([x_prev-x_cur,y_prev-y_cur])
            cos=np.sum(vec1*vec2)/np.sqrt(np.sum(vec1*vec1)*np.sum(vec2*vec2))
            out[x_cur,y_cur]=cos+1
            if i>in1 and i<out1:
                left_cosines.append(cos+1)
            else:
                right_cosines.append(cos+1)
    return out,left_cosines,right_cosines
#%%
#get those curveness scores
samples=np.random.randint(0,1200,10)
for i in samples:
    contour=contour_array[i]
    out,_1,_2=compute_cos(contour,step=30,img_size=1800,min_step=15)
    plot_result(out,contour,img_size=1800)

#calculate curveness
cosines=[]
topCos=[]
seg1s=[]
seg2s=[]
l=len(contour_array)
for contour in tqdm(contour_array):
    out,seg1,seg2=compute_cos(contour,step=30,img_size=1800,min_step=30)
    topCos.append(np.sort(out.flatten())[::-1][:60])
    cosines.append(out)
    seg1s.append(seg1)
    seg2s.append(seg2)
    #for monitor
    #per=int(i/l*100)
    #print("\r"+"["+">"*per+" "*(100-per)+"]",end="")
    

#make video
quantile=0.5
img_size=1800
def make_frame(time):
    fig=plt.figure(dpi=100)
    t=np.int(time*fps)
    cosine=cosines[t]
    head_tail_x,head_tail_y=np.where(cosine<0)
    new_cosine=cosine.copy()
    new_cosine[new_cosine<0]=0
    x,y=np.nonzero(new_cosine)
    xbar,ybar=find_centroid(contour_array[t])
    colors=np.array(np.zeros_like(x),np.float64)
    for i in range(len(x)):
        colors[i]=new_cosine[x[i],y[i]]
    plt.scatter(head_tail_x,head_tail_y,s=3,c="lemonchiffon")
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.2)#
    
    plt.plot(xbar,ybar,"ro",markersize=3,color="red")
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.title("curveness heatmap on fish contour")
    #ax.axis("off")
    out=mplfig_to_npimage(fig)
    plt.close(fig)
    return out
    
animation = VideoClip(make_frame, duration = 30)
animation.write_videofile("videos/curveness_step30_minstep30.mp4", fps=40)


#out put top N cos/curvature
#for i in tqdm(range(len(cosines))):
    #topCos[i]=np.sort(cosines[i].flatten())[::-1][:30]
#pc1 already explains more than 95% variance, so it's already enough
curveness=np.array(topCos)
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pcs=pca.fit_transform(curveness)
plt.hist(pcs)

#most points with high pc value are those with large curveness score
outliers_ind=np.where(pcs>1)[0]
import random
for i in range(10):
    ind=random.choice(outliers_ind)
    plot_result(cosines[ind],contour_array[ind],img_size=1800)

i=0
contour=contour_array[0]
#Just a function to look at how actually one step looks like in plot
def visualize_steps(curvatures,contour,img_size=600,quantile=0.5,start=0,step=30):
    head_tail_x,head_tail_y=np.where(curvatures<0)
    new_curvatures=curvatures.copy()
    new_curvatures[new_curvatures<0]=0
    x,y=np.nonzero(new_curvatures)
    xbar,ybar=find_centroid(contour)
    colors=np.array(np.zeros_like(x),np.float64)
    dists=compute_dist(contour,xbar,ybar)
    for i in range(len(x)):
        colors[i]=new_curvatures[x[i],y[i]]
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    circ=plt.Circle((xbar, ybar), radius=np.quantile(dists,quantile),linewidth=0.5, color='red',fill=False)
    ax.add_patch(circ)
    plt.scatter(head_tail_x,head_tail_y,s=3,c="lemonchiffon")
    plt.scatter(x, y, c=colors, s=3,cmap="YlGnBu", vmin=0, vmax=0.2)#
    plt.plot(xbar,ybar,"ro",markersize=3,color="red")
    #this is the three points added
    x1=np.array([contour[start][:,0],contour[(start+step)%len(contour)][:,0],contour[(start-step)%len(contour)][:,0]]) 
    y1= np.array([contour[start][:,1],contour[(start+step)%len(contour)][:,1],contour[(start-step)%len(contour)][:,1]]) 
    plt.scatter(x1,y1,s=3)
    plt.xlim(0,img_size)
    plt.ylim(0,img_size)
    plt.colorbar()
    plt.title("curveness heatmap on fish contour")
    plt.show()
 
visualize_steps(cosines[0],contour_array[0],img_size=1800,start=200)
#so 30 is quite proper for me, maybe larger if there's still lots of outlying points due to roughness
    
#%%
#the midline thing
midlines=[]
for i in tqdm(range(len(contour_array))):
    contour=contour_array[i]
    contour=contour.squeeze()
    N=len(contour)
    in1,out1,in2,out2=find_anchor(contour)
    #the index is given in a counter clockwise way
    l1=(out1-in1)%N
    l2=(out2-in2)%N
    midline=np.zeros((min(l1,l2),2),dtype=np.float64)
    for j in range(min(l1,l2)):
        midline[j]=(contour[(in1+j)%N]+contour[(out2-j)%N])/2
    midlines.append(midline)
 #this is actually not good, happens that there are small segments counted in    
def visualize_midlines(t):
    time=int(t*40)
    contour=contour_array[time]
    midline=midlines[time]
    contour=contour.squeeze()
    fig=plt.figure(dpi=100)
    x=contour[:,0]
    y=contour[:,1]
    plt.scatter(x,y,s=0.1,color="lemonchiffon")
    xmid=midline[:,0]
    ymid=midline[:,1]
    plt.scatter(xmid,ymid,s=0.1,color="blue")
    plt.xlim(0,1800)
    plt.ylim(0,1800)
    out=mplfig_to_npimage(fig)
    plt.close(fig)
    return out
    
animation = VideoClip(visualize_midlines, duration = 30)
animation.write_videofile("videos/midlines.mp4", fps=40)

l1s=[]
l2s=[]
for i in tqdm(range(len(contour_array))):
    contour=contour_array[i]
    contour=contour.squeeze()
    N=len(contour)
    in1,out1,in2,out2=find_anchor(contour,cooldown=50)
    #the index is given in a counter clockwise way
    l1=(out1-in1)%N
    l2=(out2-in2)%N
    l1s.append(l1)
    l2s.append(l2)
    
