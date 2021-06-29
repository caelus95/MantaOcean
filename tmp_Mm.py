#!/usr/bin/env python3

def read_mean4nc2(r_path,Output_name,Output_dir,skip_Total_mean=True):
    '''
    
    Inputs : str
        r_path      : Data path
        Output_name : Monthly mean data mean ex) Output_name_1993_01.nc
        Output_dir  : Output directory
        skip_Total_mean
        
    Outputs :
    
        Return a directory named 'Monthly'.
        
    '''    

    import numpy as np
    import xarray as xr
    from tqdm import tqdm
    import os 
    import shutil
    import warnings
    
    if (type(r_path) != str) or (type(Output_name) != str) or (type(Output_dir) != str) or type(skip_Total_mean)!=bool:
        raise ValueError("'r_path', 'Output_name' and 'Output_dir' should be str")
    
    if any(file == 'tmp_data' for file in os.listdir(r_path)):
        raise ValueError("'tmp_data' directory already exists in "+r_path)
    
    nc_list = np.sort([file for file in os.listdir(r_path) if file.endswith(".nc")])
    tmp_len1 = len(nc_list)
    print('\n!!!!!!!!!!!!!!!!!!\n!!!Checking data suitability...!!!\n!!!!!!!!!!!!!!!!!!')
    try :
        xr.open_dataset(r_path+nc_list[0],decode_times=True)
    except : 
        raise ValueError('Dataset does not suit for this code. No time variables in '+nc_list[0])
    print('!!!!!!!!!!!!!!!!!!\n!!!Ignore warnings for numpy...!!!\n!!!!!!!!!!!!!!!!!!')    
    warnings.filterwarnings("ignore")
    print('!!!!!!!!!!!!!!!!!!\n!!!Do not quit while processing!!!\n!!!It will mess up your directories!!!\
          \n!!!!!!!!!!!!!!!!!!')
    print('\n !!! 1 of 2 proceeding... !!!')
    os.mkdir(r_path+'tmp_data')
    os.chdir(r_path+'tmp_data')
    for i in tqdm(nc_list):
        tmp_data = xr.open_dataset(r_path+i,decode_times=True)
        tmp_name = str(tmp_data.time.values[0])[:7]
        
        if not(any(tmp_i==tmp_name for tmp_i in os.listdir())):
            os.mkdir(tmp_name)
            # os.chdir(r_path+'tmp_data/'+tmp_name)
        shutil.move(r_path+i,r_path+'tmp_data/'+tmp_name)
    
    print('\n !!! 2 of 2 proceeding... !!!')
    os.mkdir('Monthly')
    # ----------------------Slice-----------------------------------
    for j in np.sort(os.listdir()):
        print('Mean...'+j)
        tmp_dataset = xr.open_mfdataset(j+'/*.nc', parallel=True,decode_times=True)
        Data_Mmean = tmp_dataset.resample(time="1MS").mean(dim="time")
        if (j == 'Monthly') and not(skip_Total_mean):
            Data_Mmean.to_netcdf(path='./Monthly/Total.nc',mode='w')
            continue
        elif (j == 'Monthly') and skip_Total_mean :
            continue
        else :
            tmp_name4Rm = j.split('-')[0]+'_'+j.split('-')[1]
            Data_Mmean.to_netcdf(path='./Monthly/'+Output_name+'_'+tmp_name4Rm+'.nc',mode='w')
            tmp_list = os.listdir('./'+j)
            for tmp_j in tmp_list:
                shutil.move(r_path+'tmp_data/'+f"{j}/{tmp_j}",r_path)
    print('End')
    shutil.move('./Monthly',Output_dir)    
    os.chdir(r_path)
    shutil.rmtree('./tmp_data')
    tmp_len2 = len(np.sort([file for file in os.listdir(r_path) if file.endswith(".nc")]))
    if tmp_len1 != tmp_len2:
        print('File may deleted while processing !!!!')
    else :
        print('Check '+Output_dir+'/Monthly/~')
    return
        
            

    
r_path = '/home/caelus/dock_2/psi36/DATA/ncfile/GRSST/nc_files/'
Output_name = 'GRSST_monthly'
Output_dir = '/home/caelus/dock_2/psi36/DATA/ncfile/GRSST/nc_files/'
skip_Total_mean=True        
read_mean4nc2(r_path,Output_name,Output_dir)




















        
        
        
        
        
        
        
        
        

        

        
        
        
        
        
        
        
        
    
