# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:02:33 2019

@author: manta36
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:12:43 2019

@author: manta36
"""


tS = dt.date.toordinal(dt.date(int(str(dateST)[0:4]),int(str(dateST)[4:6]),int(str(dateST)[6:8])))
tE = dt.date.toordinal(dt.date(int(str(dateED)[0:4]),int(str(dateED)[4:6]),int(str(dateED)[6:8])))
d1 = pd.DataFrame(np.arange(tS,tE+1))

d2 = np.zeros([len(d1),2])

n = 0
for i in d1[0]:
    d3 = dt.datetime.fromordinal(int(i))
    d2[n,0] = d3.strftime("%Y%m%d")
    n+=1
    
d2[:,1] = np.arange(len(d1))    


# coordinates of each months 
co1 = {}

n = -1 ; m = 0 
for i in d2[:,1]:
    #print(int(i))
    n+=1
    if int(i) == len(d1)-1:
       co1[int(m)] = d2[int(i-(n)):int(i+1),1]
    elif str(d2[int(i),0])[4:6] != str(d2[int(i+1),0])[4:6]:
        co1[int(m)] = d2[int(i-(n)):int(i+1),1] 
        print(str(d2[int(i),0])[0:] )
        print(n+1)
        print(int(i-(n-1)))
        #print(str(d2[int(i),0])[4:6])
        m+=1 ; n = -1

# nanmean data

var5 = np.zeros([len(co1),at,on])
m=0
for i in co1:
    var5[m] = np.nanmean(var[int(co1[i][0]):int(co1[i][-1]),:,:],axis=0,out=None)
    m+=1 
    
vgos = var5
'''    
io.savemat('E:/psi36/DATA/matfile/global/CDS_aviso/ugos.mat', {'ugos':ugos})
'''



 u_mean = np.nanmean(var5,axis=0,out=None)
 u_anomaly = var5 - u_mean
 u_data = var5
 
 
 
 
 

 v_mean = np.nanmean(Mean_var,axis=0,out=None)
 v_anomaly = Mean_var - v_mean
 v_data = Mean_var
 
 ugos = {'u_mean' : u_mean,'u_data' : u_data,'u_anomaly' : u_anomaly}
 
 io.savemat('E:/psi36/DATA/matfile/global/CDS_aviso/ugos.mat', ugos)
 