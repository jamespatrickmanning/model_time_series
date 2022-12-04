#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:07:16 2022

@author: user
"""

import netCDF4
import datetime
import zlconversions as zl  # this is a set of Lei Zhao's functions that must be in same folder 
import numpy as np
import pandas as pd
from pandas import read_csv,DataFrame,to_datetime
from matplotlib import pyplot as plt
from dateutil.parser import parse
import conversions
def get_doppio_url(date):
    # form the doppio url based on hours since Nov 11, 2017 where "date" is a datetime
    tw=str(int((date-datetime.datetime(2017,11,1,0,0,0)).total_seconds()/3600)) # gives hours since Nov 11, 2017
    url='http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best?\
        time['+tw+':'+tw+'],h[0:1:105][0:1:241],lon_rho[0:1:105][0:1:241],lat_rho[0:1:105][0:1:241],temp['+tw+':1:'+tw+'][0:1:0][0:1:105][0:1:241]'
    return url
def get_doppio_Best(time=datetime.datetime.now(),lat=0,lon=0,depth='bottom',fortype='temperature'):#,ntimes=1):
    """
    This is JiM's simplification of ZL's get_doppio.   
    This is the newer version where we get runs from the "History_Best" single site rather than the daily files.
    outputs the temperature of point location
    fortype ='temperature' where, in the future, we might want to add "temp & depth"
    
    Modified by JiM in Nov 2022 to allow more than one point
    """
    if not doppio_coordinate(lat,lon):
        print('the lat and lon out of range in doppio')
        return np.nan,np.nan
    try:
            url=get_doppio_url(time)
            #print(url)
            nc=netCDF4.Dataset(url)
            lons=nc.variables['lon_rho'][:]
            lats=nc.variables['lat_rho'][:]
            doppio_time=nc.variables['time']
            if fortype=='depth':
                doppio_temp=nc.variables['h']
            else:
                doppio_temp=nc.variables['temp']
    except:
            print('no model data in for this time and place.')
    min_diff_time=abs(datetime.datetime(2017,11,1,0,0,0)+datetime.timedelta(hours=int(doppio_time[0]))-time)
    min_diff_index=0
    for i in range(1,len(doppio_time)):
            diff_time=abs(datetime.datetime(2017,11,1,0,0,0)+datetime.timedelta(hours=int(doppio_time[i]))-time)
            if diff_time<min_diff_time:
                min_diff_time=diff_time
                min_diff_index=i
    #calculate the min,second small and third small distance and index
    target_distance=zl.dist(lat1=lats[0][0],lon1=lons[0][0],lat2=lats[0][1],lon2=lons[0][1])
    index_1,index_2=zl.find_nd(target=target_distance,lat=lat,lon=lon,lats=lats,lons=lons)

    layer_index=0 #apparently bottom
    # if the point is on the edge of the grid, we apparently need to move it one cell away for the fitting routine to work?
    if index_1==0:
        index_1=1
    if index_1==len(lats)-1:
        index_1=len(lats)-2
    if index_2==0:
        index_2=1
    if index_2==len(lats[0])-1:
        index_2=len(lats[0])-2
    #for k in range(ntimes):
    #      point_temp_all=[]
    if fortype[0:4]=='temp':
        point=[[lats[index_1][index_2],lons[index_1][index_2],doppio_temp[min_diff_index,layer_index,index_1,index_2]],\
            [lats[index_1-1][index_2],lons[index_1-1][index_2],doppio_temp[min_diff_index,layer_index,(index_1-1),index_2]],\
            [lats[index_1+1][index_2],lons[index_1+1][index_2],doppio_temp[min_diff_index,layer_index,(index_1+1),index_2]],\
            [lats[index_1][index_2-1],lons[index_1][index_2-1],doppio_temp[min_diff_index,layer_index,index_1,(index_2-1)]],\
            [lats[index_1][index_2+1],lons[index_1][index_2+1],doppio_temp[min_diff_index,layer_index,index_1,(index_2+1)]]]
    else: # for the case of depth
        point=[[lats[index_1][index_2],lons[index_1][index_2],doppio_temp[index_1,index_2]],\
            [lats[index_1-1][index_2],lons[index_1-1][index_2],doppio_temp[(index_1-1),index_2]],\
            [lats[index_1+1][index_2],lons[index_1+1][index_2],doppio_temp[(index_1+1),index_2]],\
            [lats[index_1][index_2-1],lons[index_1][index_2-1],doppio_temp[index_1,(index_2-1)]],\
            [lats[index_1][index_2+1],lons[index_1][index_2+1],doppio_temp[index_1,(index_2+1)]]]
    point_temp=fitting(point,lat,lon)
    if np.isnan(point_temp):
        if fortype[0:4]=='temp':
            point_temp= float(doppio_temp[min_diff_index,layer_index,index_1,index_2])
        else:
            point_temp= float(doppio_temp[index_1,index_2])
    return point_temp

def fitting(point,lat,lon):
#represent the value of matrix
    ISum = 0.0
    X1Sum = 0.0
    X2Sum = 0.0
    X1_2Sum = 0.0
    X1X2Sum = 0.0
    X2_2Sum = 0.0
    YSum = 0.0
    X1YSum = 0.0
    X2YSum = 0.0

    for i in range(0,len(point)):
        
        x1i=point[i][0]
        x2i=point[i][1]
        yi=point[i][2]

        ISum = ISum+1
        X1Sum = X1Sum+x1i
        X2Sum = X2Sum+x2i
        X1_2Sum = X1_2Sum+x1i**2
        X1X2Sum = X1X2Sum+x1i*x2i
        X2_2Sum = X2_2Sum+x2i**2
        YSum = YSum+yi
        X1YSum = X1YSum+x1i*yi
        X2YSum = X2YSum+x2i*yi

#  matrix operations
# _mat1 is the mat1 inverse matrix
    m1=[[ISum,X1Sum,X2Sum],[X1Sum,X1_2Sum,X1X2Sum],[X2Sum,X1X2Sum,X2_2Sum]]
    mat1 = np.matrix(m1)
    m2=[[YSum],[X1YSum],[X2YSum]]
    mat2 = np.matrix(m2)
    _mat1 =mat1.getI()
    mat3 = _mat1*mat2

# use list to get the matrix data
    m3=mat3.tolist()
    a0 = m3[0][0]
    a1 = m3[1][0]
    a2 = m3[2][0]
    y = a0+a1*lat+a2*lon

    return y
def doppio_coordinate(lat,lon):
    f1=-0.8777722604596849*lat-lon-23.507489034447012>=0
    f2=-1.072648270137022*lat-40.60872567829448-lon<=0
    f3=1.752828434063416*lat-131.70051451008493-lon>=0
    f4=1.6986954871237598*lat-lon-144.67649951783605<=0
    if f1 and f2 and f3 and f4:
        return True
    else:
        return False
def get_depth(loni,lati,mindist_allowed=0.4):
    # routine to get depth (meters) using vol1 from NGDC
  try:  
    if lati>=40.:
        url='https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol1.nc'
    else:
        url='https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol2.nc'
    nc = netCDF4.Dataset(url).variables 
    lon=nc['x'][:]
    lat=nc['y'][:]
    xi,yi,min_dist= nearlonlat_zl(lon,lat,loni,lati) 
    if min_dist>mindist_allowed:
      depth=np.nan
    else:
      depth=nc['z'][yi,xi]
  except:
    url='https://coastwatch.pfeg.noaa.gov/erddap/griddap/srtm30plus_LonPM180.csv?z%5B(33.):1:(47.)%5D%5B(-78.):1:(-62.)%5D'  
    df=pd.read_csv(url)
    lon=df['longitude'].values[1:].astype(np.float)
    lat=df['latitude'].values[1:].astype(np.float)
    i= nearlonlat(lon,lat,loni,lati)
    depth=df['z'].values[i]
  return depth#,min_dist

def nearlonlat_zl(lon,lat,lonp,latp): # needed for the next function get_FVCOM_bottom_temp 
    """ 
    used in "get_depth"
    """ 
    # approximation for small distance 
    cp=np.cos(latp*np.pi/180.) 
    dx=(lon-lonp)*cp
    dy=lat-latp 
    xi=np.argmin(abs(dx)) 
    yi=np.argmin(abs(dy))
    min_dist=111*np.sqrt(dx[xi]**2+dy[yi]**2)
    return xi,yi,min_dist

def nearlonlat(lon,lat,lonp,latp): # needed for the next function get_FVCOM_bottom_temp
    """
    i=nearlonlat(lon,lat,lonp,latp) change
    find the closest node in the array (lon,lat) to a point (lonp,latp)
    input:
        lon,lat - np.arrays of the grid nodes, spherical coordinates, degrees
        lonp,latp - point on a sphere
        output:
            i - index of the closest node
            For coordinates on a plane use function nearxy           
            Vitalii Sheremet, FATE Project  
    """
    cp=np.cos(latp*np.pi/180.)
    # approximation for small distance
    dx=(lon-lonp)*cp
    dy=lat-latp
    dist2=dx*dx+dy*dy
    i=np.argmin(dist2)
    return i#


def getobs_tempsalt(site):
    """
    Function written by Jim Manning to get emolt data from url, return datetime, depth, and temperature.
    
    There was evidently an earlier version where a user could also specify an time range. This "input_time" can either contain two values: start_time & end_time OR one value:interval_days
    and they should be timezone aware.    example: input_time=[dt(2003,1,1,0,0,0,0,pytz.UTC),dt(2009,1,1,0,0,0,0,pytz.UTC)]
    
    Modified Nov 2022 to accept an ascii dump of entire emolt set
    """
    try:
        url = 'https://comet.nefsc.noaa.gov/erddap/tabledap/eMOLT.csvp?time,depth,sea_water_temperature&SITE=%22'+str(site)+'%22&orderBy(%22time%22)'
        df=read_csv(url,skiprows=[1])
        df['time']=df['time (UTC)']
        temp=1.8 * df['sea_water_temperature (degree_C)'].values + 32 #converts to degF
        depth=df['depth (m)'].values
        time=[];
        for k in range(len(df)):
            time.append(parse(df.time[k]))
        print('using erddap')            
    except:
        try:
            df=read_csv('../sql/eMOLT.csv',header=None,delimiter='\s+') # use this option when the ERDDAP-read_csv-method didn't work
            # see the top of emolt_notes for instructions, requires time depth temp header
            temp=df[3].values
            depth=df[2].values
            #df['time']=pd.to_datetime(df[0]+" "+df[1])
            #('converting to datetime')
            time=to_datetime(df[0]+" "+df[1])
            print('using csv file previously created for this site ')
        except:
            print('using burton & george dump file')
            df=read_csv('input/emolt_dump_nov2022.csv')
            df=df[df['SITE']==site]
            temp=df['TEMP'].values
            for kk in range(len(temp)):
                temp[kk]=temp[kk]*1.8+32
            depth=df['DEPTH'].values
            time=to_datetime(df['TIME'].values)
    dfnew=DataFrame({'temp':temp,'Depth':depth},index=time)
    return dfnew

def getemolt_latlon(site):
    """
    get lat, lon, and depth for a particular emolt site 
    """
    urllatlon = 'http://comet.nefsc.noaa.gov/erddap/tabledap/eMOLT.csvp?latitude,longitude,depth&SITE=%22'+str(site)+'%22&distinct()'
    df2=read_csv(urllatlon)
    dd=max(df2["depth (m)"])
    return df2['latitude (degrees_north)'][0], df2['longitude (degrees_east)'][0], dd

#MAIN
case='deep'#'eMOLT_site'# offshore or inshore
site='BN01'
model='DOPPIO'
year=2020
if case=='offshore':
    lat=42.950
    lon=-70.600
elif case=='inshore':
    lat=43.033
    lon=-70.683
elif case=='shallow':
    lat=43.03021
    lon=-70.71365
    depth_given=conversions.fth2m(35./6.)# converts to meters
elif case=='mid':
    lat=42.99643
    lon= -70.69723
    depth_given=conversions.fth2m(80./6.)# converts to meters
elif case=='deep':
    lat=42.94095
    lon= -70.65146
    depth_given=conversions.fth2m(160./6.)# converts to meters
else:
    [lat,lon,depthe]=getemolt_latlon(site)
numdays=365
time1=datetime.datetime(year,1,1,0,0,0)
depth=float(get_depth(lon,lat))# gets NGDC depth in meters
print('NGDC depth ='+"{0:.1f}".format(depth)+' meters.\nGiven depth ='+"{0:.1f}".format(depth_given)+' meters')
depthm=get_doppio_Best(time1,lat,lon,depth='bottom',fortype='depth')# gets model depth
dt,temp=[],[]

for k in range(numdays):
    dt.append(time1+datetime.timedelta(days=k))
    temp.append(get_doppio_Best(dt[-1],lat,lon,depth='bottom',fortype='temperature'))
    print(dt[k],temp[k])
df=pd.DataFrame([dt,temp])
df=df.T
df=df.set_index(0)
df.to_csv(case+str(time1.year)+'.csv')

#ax=plt.figure()
ax=df.plot(label=case)
ax.set_xlabel(str(year))
# add eMOLT site 
if case=='eMOLT_site':
    dfe=getobs_tempsalt(site)#returns degF
    dfe=dfe[dfe.index.year==year]
    dfe=dfe[dfe.index<max(df.index).tz_localize('utc')]
    for k in range(len(dfe)):# we have to convert back to degC
        dfe['temp'][k]=(dfe['temp'][k]-32.)/1.8
    dfe['temp'].plot(ax=ax,label=site+' ('+"{0:.1f}".format(depthe)+' meters)')
    ax.set_title(case+' '+site)
    plt.legend([model+' ('+"{0:.1f}".format(depthm)+' meters)','observed ('+"{0:.1f}".format(depthe)+' meters)'])
else:
    plt.legend([model]) 
    ax.set_title(case)
plt.savefig(case+str(time1.year)+'.png')
