import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib as mpl

DIR = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/yan_sun_eemds/'
r_path = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/latest/sigs/PDO.npy'
w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/latest/figs/'
# w_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/figs/'

# data_list = np.sort([file for file in os.listdir(DIR) if file.endswith(".npy")])

S2 = np.load(r_path)
SIG2 = pd.DataFrame(S2,columns=['data'])
SIG2['date'] = date

t = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m')#.to_period()




imf5 = pd.DataFrame(SIG1['5_mod'].values.reshape(-1)+.36,columns=['data'])

imf5 = pd.DataFrame(SIG['data'].values.reshape(-1),columns=['data']) 


date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')

imf5['date'] = date

imf5.columns = ['data','date']
# =============================================================================
# plotting
# =============================================================================


marker = 1

data = imf5.data 
doublediff = np.diff(np.sign(np.diff(data)))
peak_locations = np.where(doublediff == -2)[0] + 1

doublediff2 = np.diff(np.sign(np.diff(-1*data)))
trough_locations = np.where(doublediff2 == -2)[0] + 1


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True



plt.figure(1,figsize=(16,10),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')
plt.plot(date,SIG2['data'].values,color=[.7,.7,.7],linewidth=2,label='original')
plt.plot(date,R_m_sig.values,label='running mean',color='darkred',linewidth=2.5)
plt.plot(date,LPF,label='LPF(order=2)',color='darkblue',linewidth=2.5)
plt.plot(date,result.trend,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.plot(imf5.date,imf5.data, label='imf5(eemd)',color='k',linewidth=3.5)
if marker:
    plt.scatter(imf5.date[peak_locations], imf5.data[peak_locations],
                marker=mpl.markers.CARETUPBASE, color='darkred', s=100)
    plt.scatter(imf5.date[trough_locations], imf5.data[trough_locations],
                marker=mpl.markers.CARETDOWNBASE, color='darkblue', s=100)
    
    # for ii in trough_locations:
    #     plt.text(imf5.date[ii], imf5.data[ii]-.1, imf5.date[ii], 
    #              horizontalalignment='center', color='darkblue',fontsize=15)
    # for jj in peak_locations:
    #     plt.text(imf5.date[jj], imf5.data[jj]+.1, imf5.date[jj], 
    #              horizontalalignment='center', color='darkred',fontsize=15)

    for ii in trough_locations:
        plt.text(imf5.date[ii], -2.5, imf5.date[ii], 
                 horizontalalignment='center', color='darkblue',fontsize=15)
    for jj in peak_locations:
        plt.text(imf5.date[jj], 1.5, imf5.date[jj], 
                 horizontalalignment='center', color='darkred',fontsize=15)


# Decoration
# plt.ylim(50,750)
xtick_location = imf5.index.tolist()[::12*2]
xtick_labels = imf5.date.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper left')
plt.grid(axis='y', alpha=.3)
plt.savefig(w_path1+'pdo_decompo')
plt.show()







# =============================================================================
# 
# =============================================================================




