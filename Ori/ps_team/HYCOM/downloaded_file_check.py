

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:05:46 2020

@author: manta36
"""


import os
import numpy as np
import datetime as dt

Directory = 'D:/Data/mat/HYCOM/D'
file_start ='20000101'
file_end = '20101231'


ft = dt.date.toordinal(dt.date(int(file_start[:4]),int(file_start[4:6]),int(file_start[6:8])))
fd = dt.date.toordinal(dt.date(int(file_end[:4]),int(file_end[4:6]),int(file_end[6:8])))

date_list = list(np.arange(ft,fd+1))

n = 0
for i in date_list:
    date_list[n] = int(str(dt.date.fromordinal(i)).replace('-',''))
    n+=1
##########################

dir_list = os.listdir(Directory)
nc_list = [file.split('_')[1].replace('.mat','') for file in dir_list if file.endswith(".mat")]
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


