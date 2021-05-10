# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:05:01 2019

@author: manta36
"""


data = var_dic['data']

k,l,r = np.shape(data)

date = np.arange(300)

DJF = np.full([int(k/12)+1,l,r], np.nan)
DJF_temp = np.full([3,l,r], np.nan)
m = 0
n = 0 
for i in date :
    #print(i%12)
    if i%12 == 0 or i%12 ==  1 or i%12 == 11 :
        DJF_temp[m,:,:] = data[i,:,:]
        m+=1
        #print('v')
        if i%12 == 1  :
            #print('w')
            DJF[n,:,:] = np.nanmean(DJF_temp,axis=0,out=None)
            m = 0 ; n +=1
            DJF_temp = np.full([3,l,r], np.nan)
        elif i == k-1 and i%12 == 11:
            DJF[n,:,:] = np.nanmean(DJF_temp,axis=0,out=None)
            
        
var_dic1={'DJF':DJF}

'''
# save as mat(matlab) format
# double ==> 
io.savemat('F:/DATA/matfile/test/np_vector2.mat', {'var_dic':var_dic})
#struct ==> 
io.savemat('E:/DATA/matfile/np_vector.mat', var_dic1)
'''

