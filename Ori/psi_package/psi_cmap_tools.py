#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:16:27 2020

@author: shamu
"""

def psi_cm_mat2npz(matname,npzname):
    
    # matname = '/home/shamu/Wormhole/othercolor/colorData.mat'
    # npzname = '/home/shamu/Wormhole/tmp_np/cm_1'
        
    from scipy import io
    import numpy as np
    
    tmp = io.loadmat(matname)
    keys_list = list(tmp.keys())[3:]
    M = {}
    for i in keys_list :
        cm_matrix = tmp[i]
        M[i] = cm_matrix
        print(i)
    np.save(npzname,M)
    return keys_list


def psi_get_cm(cmap_name,div_N,cmap_N=1,DIR='',test=False) :

    import numpy as np
    import matplotlib.pyplot as plt 
    from matplotlib.colors import LinearSegmentedColormap 

    if not DIR:
        Dir = '/home/shamu/mangrove1/Working_hub/LGnDC_dep/python_cent/pycode_team/plotting/cmaps/'
        print('Current colormap directory is :\n'+Dir)
    elif DIR:
        Dir = DIR
        print('Change colormap directory to :\n'+Dir)

    import os 
    
    temp_list = [i for i in os.listdir(Dir) if i.endswith('.npy')]
    print('\n!!!___cmap___!!!\n')
    for i,j in zip(range(1,len(temp_list)+2),temp_list):
        print(str(i)+' : '+str(j))
    print('\nCurrent cmap = '+str(cmap_N))
    
    Cmap_pkg = np.load(Dir+temp_list[cmap_N-1],allow_pickle=True)
    
    cm_matrix = Cmap_pkg.item().get(cmap_name)
    
    # cm_matrix.item().get('BuGy_8')
    # cm_matrix.item().keys()
    
    cm = LinearSegmentedColormap.from_list(cmap_name,cm_matrix,N=div_N) 

    if test:
        # Make some illustrative fake data: 
        a = np.arange(0, np.pi, 0.1) 
        b = np.arange(0, 2 * np.pi, 0.1) 
        A, B = np.meshgrid(a, b) 
        X = np.cos(A) * np.sin(B) * 10
        # Discretizes the interpolation into bins 
        all_bins = [10, 15, 25, 50] 
        figure, axes = plt.subplots(2, 2,figsize =(6, 9)) 
        figure.subplots_adjust(left = 0.02,bottom = 0.06,right = 0.95, 
        					top = 0.94, wspace = 0.05) 
        for all_bin, ax in zip(all_bins, axes.ravel()): 
        	# Making the the colormap 
            	cm_test = LinearSegmentedColormap.from_list(
            		cmap_name,cm_matrix,N = all_bin) 
            	
            	im = ax.imshow(X, interpolation ='nearest', 
            				origin ='lower', cmap = cm_test) 
            	
            	ax.set_title("bin: % s" % all_bin) 
            	figure.colorbar(im, ax = ax) 
    return cm
                
            