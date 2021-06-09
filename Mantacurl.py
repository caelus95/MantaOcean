#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:21:22 2021

@author: caelus
"""


def MantaCurl2D(u,v,dx=28400.0,dy=28400.0 ):
    import numpy as np
    '''
    dx = 28400.0 # meters calculated from the 0.25 degree spatial gridding 
    dy = 28400.0 # meters calculated from the 0.25 degree spatial gridding 
    '''
    dv_dx, dv_dy = np.gradient(u, dx,dy)
    du_dx, du_dy = np.gradient(v, dx,dy)

    curl = dv_dx - du_dy
    return curl





