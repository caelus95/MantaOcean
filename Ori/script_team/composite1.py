# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:29:27 2019

@author: navys
"""


def composite1(var_name):
    
    
    '''
    1993년 시작 기준 
    F:/psi36/DATA/matfile/composite_years/kuroshio_sshdif 참고
    '''


    import numpy as np
    from scipy import io
    import os

    # var = io.loadmat('F:/psi36/DATA/temp_var3/EastP_ugos')

    period_Directory = 'F:/psi36/DATA/matfile/composite_years/'
    mat_name = 'kuroshio_sshdif'
    #var_name = var_dic['anomaly']

    dir_list=os.listdir(period_Directory)
    mat_list=[file for file in sorted(dir_list) if file.endswith(mat_name+'_hyr_lyr.mat')]

    period = io.loadmat(period_Directory+mat_list[0])

    composite_list=[period['hyr'],period['Lyr']]

    l,s = np.shape(var_name[0,:,:])[0],np.shape(var_name[0,:,:])[1]
    k=0
    for i in composite_list:
        var=i
        print('a')
        print(i)
        n=1 ; m=0
        temp_composite= np.zeros((int(len(var[1,:])/2),l,s))
        while n <= len(var[1,:]):
                temp_composite[m,:,:] = np.nanmean(var_name[var[0,n]:var[-1,n]+1,:,:],axis=0,out=None)
                n+=2 ; m +=1
        if k==0:
            var_composite = {'hyr':temp_composite}
        
        else:
            var_composite['lyr'] = temp_composite            
        k+=1
        
    return var_composite

'''
# save as mat(matlab) format
# double ==> 
io.savemat('F:/DATA/matfile/test/np_vector2.mat', {'var_composite':var_composite})
#struct ==> 
io.savemat('E:/psi36/DATA/temp_var/ET_vgos_composite.mat', var_composite)
'''
        