#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:07:14 2021

@author: caelus
"""

def Manta_Windstress(u,v):
    '''
    % Tau=Rho*Cd*(speed)^2; Tx=Rho*Cd*Speed*u; Ty=Rho*Cd*Speed*v
    %===========================================================%
    % RA_WINDSTR  $Id: ra_windstr.m, 2014/10/29 $
    %          Copyright (C) CORAL-IITKGP, Ramkrushn S. Patel 2014.
    %
    % AUTHOR: 
    % Ramkrushn S. Patel (ramkrushn.scrv89@gmail.com)
    % Roll No: 13CL60R05
    % Place: IIT Kharagpur.
    % This is a part of M. Tech project, under the supervision of DR. ARUN CHAKRABORTY
    %===========================================================%
    %
    % USAGE: [Tx, Ty]=ra_windstr(u,v)
    %  
    % DESCRIPTION:  Function to compute wind stress from wind field data Based on Gill, 1982 
    % Formula and a non-linear Cd based on Large and Pond (1981), modified for low wind 
    % speeds (Trenberth et al., 1990)
    % 
    % INPUTS: 
    % u = Zonal wind component [m/s], must be 2D
    % v = Meridional wind component [m/s], must be 2D
    %
    % OUTPUT: 
    % Tx = Zonal wind stress [N/m^2]
    % Ty = Meridional wind stress [N/m^2]
    % 
    % DISCLAIMER: 
    % Albeit this function is designed only for academic purpose, it can be implemented in 
    % research. Nonetheless, author does not guarantee the accuracy.
    % 
    % REFERENCE:
    % A.E. Gill, 1982, Atmosphere-Ocean Dynamics, Academy Press, Vol. 30.
    % W. G. Large & S. Pond., 1981,Open Ocean Measurements in Moderate to Strong Winds, 
    % J. Physical Oceanography, Vol. 11, pp. 324 - 336.
    % K.E. Trenberth, W.G. Large & J.G. Olson, 1990, The Mean Annual Cycle in Global Ocean 
    % Wind Stress, J.Physical Oceanography, Vol. 20, pp. 1742  1760.
    %
    % ACKNOWLEDGMENT:
    % Author is eternally grateful to MathWorks for providing in built functions. 
    % ***********************************************************************************************%
    '''
    
    #Air density
    roh=1.2 # kg/m**2
    
    # Computation of wind Stresses
    lt, ln = np.shape(u)
    Tx = np.zeros([lt, ln])
    Ty = np.zeros([lt, ln])
    
    Tx[Tx==0] = np.nan
    Ty[Ty==0] == np.nan
    
    for i in range(lt):
        for j in range(ln):
            U = np.sqrt(u[i,j]**2+v[i,j]**2)
            if U <= 1:
                Cd = 0.00218
            elif (U > 1) or (U <= 3):
                Cd = (0.62+1.56/U)*.001
            elif (U > 3) or (U < 10):
                Cd = .00114
            else :
                Cd=(0.49+0.065*U)*.001

            Tx[i,j] = Cd*roh*U*u[i, j]
            Ty[i,j] = Cd*roh*U*v[i, j]
            
    return Tx, Ty
        

def Manta_WindStressCurl(lat,lon,u,v):
    '''  
    % curlZ = dTy/dx - dTx/dy; solved by finite difference method
    %===========================================================%
    % RA_WINDSTRCURL  $Id: ra_windstrcurl.m, 2015/01/15 $
    %          Copyright (C) CORAL-IITKGP, Ramkrushn S. Patel 2014.
    %  
    % AUTHOR: 
    % Ramkrushn S. Patel (ramkrushn.scrv89@gmail.com)
    % Roll No: 13CL60R05
    % Place: IIT Kharagpur.
    % This is a part of M. Tech project, under the supervision of DR. ARUN CHAKRABORTY
    %===========================================================%
    %
    % USAGE: curlz=ra_windstrcurl(lat,lon,u,v)
    %  
    % PREREQUISITE:
    % ra_windstr.m written by same author
    %
    % DESCRIPTION:  Function to compute wind stress curl from wind field data Based on 
    % NRSC (2013)
    % 
    % INPUTS:
    % lat = Latitude vector [deg.]
    % lon = Longitude Vector [deg.]
    % u = Zonal wind component [m/s], must be 2D
    % v = Meridional wind component [m/s], must be 2D
    %
    % OUTPUT: 
    % curlZ = Wind stress curl [N/m^3]
    % 
    % DISCLAIMER: 
    % Albeit this function is designed only for academic purpose, it can be implemented in 
    % research. Nonetheless, author does not guarantee the accuracy.
    % 
    % REFERENCE:
    % A.E. Gill, 1982, Atmosphere-Ocean Dynamics, Academy Press, Vol. 30.
    % W. G. Large & S. Pond., 1981,Open Ocean Measurements in Moderate to Strong Winds, 
    % J. Physical Oceanography, Vol. 11, pp. 324 - 336.
    % K.E. Trenberth, W.G. Large & J.G. Olson, 1990, The Mean Annual Cycle in Global Ocean 
    % Wind Stress, J.Physical Oceanography, Vol. 20, pp. 1742  1760.
    % NRSC, 2013, "OSCAT Wind stress and Wind stress curl products", Ocean Sciences Group,
    % Earth and Climate Science Area, Hyderabad, India.
    %
    % ACKNOWLEDGMENT:
    % Author is grateful to MathWorks for developing in built functions. 
    '''
    
    # Degress to radian
    rad = np.pi/180
    # Wind Stresses computaion
    [Tx,Ty]=Manta_Windstress(u, v);
    
    # Computation of curl
    lt, ln = np.shape(u)
    a = np.diff(lat)
    aa = np.nan*np.zeros([len(a)-1,1])

    for i in range(len(a)-1):
        if a[i] == a[i+1]:
            aa[i] = a[i]
        else: 
            print("Unexpected error:", sys.exc_info()[0])
            raise
        dlat=np.mean(aa)
        
    deltay = dlat*111176
    curlZ,long = np.zeros([lt, ln]),np.zeros([lt, ln])
    curlZ[curlZ==0] = np.nan
    long[long==0] == np.nan
    for i in range(lt):
        for j in range(ln):
            long[i,j] = lon[j]*111176*np.cos(lat[i]*rad)
            # long[i,j] = lon[j]*6378137*rad*np.cos(lat[i]*rad)
            # [m] earth radious in meters= 6,378,137.0 m.. from wikipedia.


    # Centeral difference method in x and y
    for i in range(1,lt-1):
        for j in range(1,ln-1):
            curlZ[i,j] = (Ty[i, j+1]-Ty[i, j-1])/(2*(long[i, j+1]-long[i, j-1])) - (Tx[i+1, j]-Tx[i-1, j])/(2*deltay) 
        
    # Forward difference method in x and y 
    for j in range(ln-1) :
        curlZ[0, j]=(Ty[0, j+1]-Ty[0, j])/(long[0, j+1]-long[0, j]) - (Tx[1, j]-Tx[0, j])/deltay 

    for i in range(lt-1):
        curlZ[i,0] = (Ty[i, 1]-Ty[i, 0])/(long[i, 1]-long[i, 0]) - (Tx[i, 1]-Tx[i, 0])/deltay 

    curlZ[0, ln-1] = curlZ[0, ln-2]

    # Backward difference method in x and y

    for i in range(1,lt):
        curlZ[i, ln-1] = (Ty[i, ln-1]-Ty[i, ln-2])/(long[i, ln-1]-long[i, ln-2]) - (Tx[i, ln-1]-Tx[i-2, ln-1])/deltay 
    for j in range(1,ln-1):
        curlZ[lt-1, j] = (Ty[lt-1, j]-Ty[lt-1, j-1])/(long[lt-1, j]-long[lt-1, j-1]) - (Tx[lt-1, j]-Tx[lt-2, j])/deltay    
        
    curlZ[lt-1, 0]=curlZ[lt-1, lt-2]

    return curlZ







