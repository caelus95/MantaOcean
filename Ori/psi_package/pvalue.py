#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:04:38 2020

@author: shamu
"""
def Pvalue(Vari,Sampling_n,Compo_n):
    from scipy.stats import norminvgauss
    import numpy as np
    from tqdm import tqdm
    # import random
    # random.seed('')
    t,idx,idy = Vari.shape
    R_Vari = np.zeros([Sampling_n,idx,idy])    

    x_Mean = np.zeros([idx,idy])
    x_Var =  np.zeros_like(x_Mean)
    x_Norminv = np.zeros([2,idx,idy])
    print('Processing...(1/2)')
    for i in tqdm(range(Sampling_n)):
        for j in range(idx):
            for k in range(idy):
                R_num = np.random.randint(low=0,high=t,size=Compo_n)
                R_Vari[i,j,k] = np.squeeze( np.nanmean(Vari[R_num,j,k]) )
    print('Processing...(2/2)')
    for i in range(idx):
        for j in range(idy):
            x_Mean[i,j] = np.nanmean(np.squeeze(R_Vari[:,i,j]))
            x_Var[i,j] = np.sqrt(np.squeeze(R_Vari[:,i,j]).var())
            x_Norminv[:,i,j] = norminvgauss.ppf([.5,.95],x_Mean[i,j],x_Var[i,j])
    return x_Mean, x_Var, x_Norminv
    