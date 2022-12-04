#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 09:35:37 2022

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
#proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
#os.environ['PROJ_LIB'] = 'c:\\Users\\Joann\\anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\share'
#os.environ['PROJ_LIB'] = 'C:\\Users\\james.manning\\Anaconda3\\pkgs\\proj-7.1.0-h7d85306_1\\Library\share'
os.environ['PROJ_LIB'] = '/home/user/anaconda3/pkgs/proj-8.2.1-h277dcde_0/share'
from mpl_toolkits.basemap import Basemap
#HARDCODES
area='IsleShoals'
lats=[42.950,43.033,43.03021,42.99643,42.94095]
lons=[-70.60,-70.683,-70.71365,-70.69723,-70.65146]
sta=['offsh','insh','shal','mid','deep']
def getgbox(area):
  # gets geographic box based on area
  if area=='SNE':
    gbox=[-72.,-65.,38.,41.5] # for SNE shelf east
  elif area=='SNW':
    gbox=[-71.5,-69.5,40.,41.75] # for SNw shelf west
  elif area=='MABN':
    gbox=[-73.,-68.,39.,42.] # for SNw shelf west  
  elif area=='OOI':
    gbox=[-71.5,-69.,39.5,41.6] # for OOI
  elif area=='GBANK':
    gbox=[-71.,-66.,40.,42.] # for GBANK
  elif area=='GBANK_RING':
    gbox=[-71.,-65.,39.,42.5] # for typical GBANK Ring 
  elif area=='GS':           
    gbox=[-71.,-63.,38.,42.5] # for Gulf Stream
  elif area=='NorthShore':
    gbox=[-71.,-69.5,41.75,43.25] # for north shore
  elif area=='IsleShoals':
    gbox=[-70.85,-70.5,42.75,43.25] # for Isle of Shoals  
  elif area=='Gloucester':
    gbox=[-71.,-70.,42.25,43.] # for north shore
  elif area=='IpswichBay':
    gbox=[-71.,-70.,42.5,43.] # for IpswitchBay
  elif area=='CCBAY':
    gbox=[-70.75,-69.8,41.5,42.23] # CCBAY
  elif area=='inside_CCBAY':
    gbox=[-70.75,-70.,41.7,42.15] # inside CCBAY
  elif area=='NEC':
    gbox=[-69.,-64.,39.,43.5] # NE Channel
  elif area=='NE':
    gbox=[-76.,-66.,35.,44.5] # NE Shelf 
  return gbox

fig = plt.figure(figsize=(10,6))
gb=getgbox(area)
fig, ax = plt.subplots()
m = Basemap(projection='stere',lon_0=(gb[0]+gb[1])/2.,lat_0=(gb[2]+gb[3])/2.,lat_ts=0,llcrnrlat=gb[2],urcrnrlat=gb[3],\
                llcrnrlon=gb[0],urcrnrlon=gb[1],rsphere=6371200.,resolution='f',area_thresh=100)# JiM changed resolution to "c" for crude
# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.fillcontinents(color='gray',zorder=3)
# draw parallels.
if (area=='IpswichBay') or (area=='Gloucester'):
    labint=0.2
    dept_clevs=[30,50,100, 150]
elif area[0:3]=='CCB':
    labint=0.5
    dept_clevs=[30,50,100]
elif (area=='NE') or (area=='SNW'):
    labint=1.0
    dept_clevs=[50,100,1000]
    x,y=m(-69.5,40.5)    
    #plt.text(x,y,'Great South Channel',fontsize=12, rotation=0) 

elif (area=='NorthShore') or (area=='IsleShoals'):
    labint=.20
    dept_clevs=[50,100,150]  
elif (area=='GBANK') or (area=='GBANK_RING'):
    labint=1.0
    dept_clevs=[50,100,150]
    x,y=m(-68.3,40.65)    
    plt.text(x,y,' Georges Bank',fontsize=16, rotation=30) 
else:
    labint=1.0
    dept_clevs=[30,50,100, 150,300,1000]
parallels = np.arange(0.,90,labint)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(180.,360.,labint)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
x, y = m(lons, lats)
ax.scatter(x,y)
#m.text(x,y,sta)
#for (x,y), label in zip(xy, sta):
#    ax.text(x, y, label, ha='center', size=10)
color='b';va='bottom' # case for first setup
for k in range(len(x)):
    if k>1:
        color='r'
        va='top'
    ax.text(x[k],y[k],sta[k],color=color,verticalalignment=va)
plt.title('Watson et al site locations',fontsize=10,fontweight='bold')
#plt.suptitle(model+' at '+time_str, fontsize=24)