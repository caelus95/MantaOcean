

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:05:46 2020

@author: manta36
"""
from netCDF4 import Dataset
import numpy as np
import datetime as dt
import time


Directory = 'D:/Data/mat/HYCOM/D/'
file_start ='20000101'
file_end = '20101231'

# slicing grid
llcrnrlat = -10 # int
urcrnrlat = 45 # int
llcrnrlon = 112 # int
urcrnrlon = 170 # int
llcrnrdepth = 0 # int
urcrnrdepth = 4000 # int

# url
url = 'dods://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/'+file_start[0:4]

# find time coor
Yorig_HYCOMa = 2000
Morig_HYCOMa = 1
Dorig_HYCOMa = 1

# save dir (create nc)  
nc_savedir = Directory # str

# functions

def FindMissingData(Directory,file_start,file_end):

    import os
    import numpy as np
    import datetime as dt


    ft = dt.date.toordinal(dt.date(int(file_start[:4]),int(file_start[4:6]),int(file_start[6:8])))
    fd = dt.date.toordinal(dt.date(int(file_end[:4]),int(file_end[4:6]),int(file_end[6:8])))

    date_list = list(np.arange(ft,fd+1))

    n = 0
    for i in date_list:
        date_list[n] = int(str(dt.date.fromordinal(i)).replace('-',''))
        n+=1
##########################

    dir_list = os.listdir(Directory)
    nc_list = [file.split('_')[1].replace('.nc','') for file in dir_list if file.endswith(".nc")]
    nc_date_list = list(np.zeros(fd-ft+1))
    missing_list = list(np.zeros(fd-ft+1-len(nc_list)))

    m,n,l = 0,0,0
    for i in date_list:
        #print('a')
        #temp_l = len(temp_var)
        if i != int(nc_list[n]):
            nc_date_list[m] = np.nan
            missing_list[l] = i
            print('Missing_value at : '+str(i))
            l+=1
        else :
            nc_date_list[m] = i
            n+=1
        m+=1
        print('!!!END / check variable missing_list!!!')
    return missing_list, nc_date_list
###########################################################

def datenum2datevec(numarange,ori_num=0):
    '''
    ori_num =
    '''
    import numpy as np
    import datetime as dt

    timevec = np.zeros([len(numarange),3])
    time_h = [int(i) for i in numarange]
    timedt = [dt.datetime.fromordinal(i + ori_num) for i in time_h]
    timelist = [[i.year,i.month,i.day] for i in timedt]
    i=0
    while i <= len(numarange)-1:
        timevec[i][:] = timelist[i]
        i+=1
    return timevec


def YunYearsdates(iyear,imonth):
    '''

    '''
    if imonth ==1 or imonth==3 or imonth==5 or imonth==7 or imonth==8 or imonth==10 or imonth==12 :
        day = 31
    elif imonth == 4 or imonth == 6 or imonth == 9 or imonth == 11:
        day = 30
    elif imonth == 2:
        if iyear%4 == 0:
            day = 29
        else:
            day = 28
    return day

def DownloadingData(time,depth_co,lat_co,lon_co):
    import numpy as np

    ts,te,deps,depe = time[0],time[-1],depth_co[0],depth_co[-1]
    lats,late,lons,lone = lat_co[0],lat_co[-1],lon_co[0],lon_co[-1]

    ssh = np.nanmean( Dataset(url+'?surf_el[%d:%d][%d:%d][%d:%d]'%(ts,te,lats,late,lons,lone),'r')['surf_el'][:] , axis=0)
    print('ssh...')
    u = np.nanmean( Dataset(url+'?water_u[%d:%d][%d:%d][%d:%d][%d:%d]'%(ts,te,deps,depe,lats,late,lons,lone),'r')['water_u'][:] , axis=0)
    print('u...')
    v = np.nanmean( Dataset(url+'?water_v[%d:%d][%d:%d][%d:%d][%d:%d]'%(ts,te,deps,depe,lats,late,lons,lone),'r')['water_v'][:] , axis=0)
    print('v...')
    temp = np.nanmean( Dataset(url+'?water_temp[%d:%d][%d:%d][%d:%d][%d:%d]'%(ts,te,deps,depe,lats,late,lons,lone),'r')['water_temp'][:] , axis=0)
    print('temperature...')
    sal = np.nanmean( Dataset(url+'?salinity[%d:%d][%d:%d][%d:%d][%d:%d]'%(ts,te,deps,depe,lats,late,lons,lone),'r')['salinity'][:] , axis=0)
    print('salinity...')
    return ssh, u, v, temp, sal

def trans2nc(nc_savedir,time_array,ndepth,nlat,nlon,nssh,nu,nv,ntemp,nsal):

    from netCDF4 import Dataset
    from numpy import dtype

    # open a netCDF file to write
    name = f"{int(time_array[0]):02d}"+f"{int(time_array[1]):02d}"+f"{int(time_array[2]):02d}"
    ncout = Dataset(nc_savedir+'HYCOM'+name+'.nc', 'w', format='NETCDF4')

    # define axis size
    #ncout.createDimension('time', None)  # unlimited
    ncout.createDimension('depth', len(ndepth))
    ncout.createDimension('lat', len(nlat) )
    ncout.createDimension('lon', len(nlon) )

    # create latitude axis
    lat = ncout.createVariable('lat', dtype('double').char, ('lat'))
    lat.standard_name = 'latitude'
    lat.long_name = 'latitude'
    lat.units = 'degrees'
    lat.axis = 'Y'

    # create longitude axis
    lon = ncout.createVariable('lon', dtype('double').char, ('lon'))
    lon.standard_name = 'longitude'
    lon.long_name = 'longitude'
    lon.units = 'degrees'
    lon.axis = 'X'

    # create variable array
    sshout = ncout.createVariable('ssh', dtype('double').char, ('lat', 'lon'))
    sshout.long_name = 'Surf_el'
    sshout.units = 'm'
    sshout.missing_value = np.nan

    uout = ncout.createVariable('u', dtype('double').char, ('depth', 'lat', 'lon'))
    uout.long_name = 'water_u'
    uout.units = 'm/s'
    uout.missing_value = np.nan

    vout = ncout.createVariable('v', dtype('double').char, ('depth', 'lat', 'lon'))
    vout.long_name = 'water_v'
    vout.units = 'm/s'
    vout.missing_value = np.nan

    tempout = ncout.createVariable('temp', dtype('double').char, ('depth', 'lat', 'lon'))
    tempout.long_name = 'water_temp'
    tempout.units = 'C'
    tempout.missing_value = np.nan

    salout = ncout.createVariable('sal', dtype('double').char, ('depth', 'lat', 'lon'))
    salout.long_name = 'water_sal'
    salout.units = 'psu'
    salout.missing_value = np.nan

    # copy axis from original dataset
    lon[:] = nlon[:]
    lat[:] = nlat[:]

    sshout[:] = nssh[:]
    uout[:] = nu[:]
    vout[:] = nv[:]
    tempout[:] = ntemp[:]
    salout[:] = sal[:]

    # close files
    ncout.close()

    return


def percentage2(i,L):
    '''
    info
    i = 0
    i = percentage2(i,L)
    '''
    import time
    global t
    j = i/L*100
    print('Processing...... %.2f'%j+'%')
    if i ==0:
        print('Calulating time expected...')
        t = time.time()
    elif i>=1:
        elapsed = (time.time() - t)/(i+1)
        print( 'Expacted Time ==> ',round(elapsed*(L-i),2),'(s)')
    i+=1
    return i

####################################################################################################
###
####################################################################################################


nc_data = HYCOMa_time=Dataset(url,'r')

print(nc_data)
print(nc_data.variables)

# slicing HYCOM time depth lat lon
HYCOM_time=Dataset(url+'?time','r')['time'][:].data
HYCOM_depth=Dataset(url+'?depth','r')['depth'][:].data

HYCOM_lat = Dataset(url+'?lat','r')['lat'][:].data
HYCOM_lon = Dataset(url+'?lon','r')['lon'][:].data

HYCOM_lat_co = np.where((HYCOM_lat >= llcrnrlat) & (HYCOM_lat <= urcrnrlat))[0]
HYCOM_lon_co = np.where((HYCOM_lon >= llcrnrlon) & (HYCOM_lon <= urcrnrlon))[0]
HYCOM_depth_co = np.where((HYCOM_depth >= llcrnrdepth) & (HYCOM_depth <= urcrnrdepth))[0]

HYCOM_lat_rgnl = HYCOM_lat[HYCOM_lat_co]
HYCOM_lon_rgnl = HYCOM_lon[HYCOM_lon_co]
HYCOM_depth_rgnl = HYCOM_depth[HYCOM_depth_co]

# fromordinal --> datevec

ori_num = dt.date.toordinal( dt.date(Yorig_HYCOMa,Morig_HYCOMa,Dorig_HYCOMa))
HYCOM_timevec = datenum2datevec(HYCOM_time/24,ori_num)

#

missing_list, nc_data_list = FindMissingData(Directory, file_start, file_end)
#
Script_terminate_n = 0
while len(missing_list) > 60:

    # calculating dates
    per_var = 0 , error_n = 1
    T_num =len(missing_list)
    for i in missing_list:
        Y,M,D = int(i[0:4]),int(i[4:6]),int(i[6,8])

        print_timevec = 

        per_var = percentage2(per_var,T_num)
        time_co = np.where( (HYCOM_timevec[:,2]==D) & (HYCOM_timevec[:,1]==M) & (HYCOM_timevec[:,0]==Y) )[0]
        # Downloading

        while True:
            if error_n == 100:
                error_n = 1
                stop_code = True
                break
            try:
                ssh, u, v, temp, sal = DownloadingData(time_co,HYCOM_depth_co,HYCOM_lat_co,HYCOM_lon_co)
                error_n, stop_code = 1, False
                break
            except:
                print('Download error occured!!! Retring...')
                #time.sleep(3)
                print('This is %d th tring...'%error_n)
                # print('This script will automatically terminate after the 10th attempt')
                error_n+=1
        if stop_code:
            continue

        trans2nc(nc_savedir,print_timevec[per_var-1],HYCOM_depth_rgnl,HYCOM_lat_rgnl,HYCOM_lon_rgnl,ssh,u,v,temp,sal)
        print('!!!END!!! '+dt.date(int(print_timevec[per_var-1,0]), int(print_timevec[per_var-1,1]), int(print_timevec[per_var-1,2])).isoformat())

    if Script_terminate_n == 30:
        print('error 이번 년도는 답도 없습니다')
        raise SystemExit


    missing_list, nc_date_list = FindMissingData(Directory, file_start, file_end)
                                                                 