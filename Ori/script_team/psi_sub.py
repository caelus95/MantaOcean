



Ac,Bc = np.meshgrid(lon_rgnl,lat_rgnl)


u_s,v_s,Acp,Bcp = subtract_vec(np.nanmean(ugos,axis=0),np.nanmean(vgos,axis=0),Ac,Bc,3)


def subtract_vec(u,v,Ac,Bc,interval) :

    '''
    % [lonr, latr] = meshgrid(lon,lat)
    % Ac =  latr(lat_co,lon_co) ;
    % Bc =  lonr(lat_co,lon_co ) ;
    %  u_2D_vec = 2x1 cell , u_2D_vec{1}(:,:) = hyr_mean / 
    %  u_2D_vec{2}(:,:) = lyr_mean;
    %  v_2D_vec = 2x1 cell , v_2D_vec{1}(:,:) = hyr_mean / 
    %  v_2D_vec{2}(:,:) = lyr_mean;

    '''
    import numpy as np

    
    [s,m] = np.shape(u) ; 

    i = np.arange(0,s,interval)
    j = np.arange(0,m,interval)
    
    #ii = np.arange(0,int(s/interval) + 1)
    #jj = np.arange(0,int(m/interval) + 1 )

    u_s = np.zeros([len(i),len(j)])
    v_s = np.zeros([len(i),len(j)])
    Acp = np.zeros([len(i),len(j)])
    Bcp = np.zeros([len(i),len(j)])
    
    ii,jj=0,0
    for i1 in i :
        for j1 in j :

            u_s[ii,jj] = u[i1,j1]    
            v_s[ii,jj] = v[i1,j1]  
            Acp[ii,jj] = Ac[i1,j1]  
            Bcp[ii,jj] = Bc[i1,j1]
            print(i1,j1,ii)
            jj+=1
        ii+=1
        jj=0
    return u_s, v_s, Acp, Bcp
    
    