#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:58:31 2020

@author: shamu
"""



def psi_save_grid_data_zip(Data_Name,lat_rgnl,lon_rgnl,lat_co,lon_co,
                           lat_F,lon_F,Data,method='np'):
    data = {
    'lat_rgnl': lat_rgnl,
    'lon_rgnl': lon_rgnl,
    'lat_co': lat_co,
    'lon_co': lon_co,
    'lat_F': lat_F,
    'lon_F':lon_F,
    'Data' : Data
    }    

    if method == 'np' :
        import numpy as np
        Name_Vari = Data_Name+'.npy'
        np.save(Data_Name,data)
    elif method == 'pickle':
        import pickle
        Name_Vari = Data_Name+'.pickle'
        with open(Name_Vari, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    elif method == 'mat':
        from scipy import io
        Name_Vari = Data_Name+'.mat'
        io.savemat(Name_Vari, data)
    print('Vari_name : '+Name_Vari)
    


def psi_load_grid_data_zip(Data_Name):
    if Data_Name.split('.')[-1] == 'npy' :
        import numpy as np
        Data_temp = np.load(Data_Name)
    elif Data_Name.split('.')[-1] == 'pickle':
        import pickle
        with open('/home/shamu/HUB2/data.pickle', 'rb') as f:
            Data_temp = pickle.load(f)
    elif Data_Name.split('.')[-1] == 'mat':
        from scipy import io
        Data_temp = io.loadmat(Data_Name)
    lat_rgnl = Data_temp['lat_rgnl']
    lon_rgnl = Data_temp['lon_rgnl']
    lat_co = Data_temp['lat_co']
    lon_co = Data_temp['lon_co']
    lat_F = Data_temp['lat_F']
    lon_F = Data_temp['lon_F']
    Data = Data_temp['Data']
    # Maxlat = lat_rgnl[-1]
    # Minlat = lat_rgnl[0]
    # Maxlon = lon_rgnl[-1]
    # Minlon = lon_rgnl[0]
    return lat_rgnl,lon_rgnl,lat_co,lon_co,lat_F,lon_F,Data
 

