


import numpy as np

import sys 
sys.path.append('/home/caelus/dock_1/Working_hub/LGnDC_dep/python_cent/MantaPKG/')

from Manta_Signals.procc_index import sig_pro, linearRegress4Cube



r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'


Sig_set,Corr_map = sig_pro(r_path,['1993-01-01',324,300],Standard=True)


dataset = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/data/adt_10_30_120_250.npy')


Coef, p_values = linearRegress4Cube(Sig_set.first_5_2Y_Rm.dropna()[:-1],dataset[12:-12,:,:],['1994-01','2018-12'],method='sm')


Coef, p_values = linearRegress4Cube(Sig_set.first_5_ceemd_imf5,dataset,['1993-01','2019-12'],method='sm')


import matplotlib.pyplot as plt

plt.pcolormesh(Coef)
plt.colorbar()
plt.clim([-10,10])








