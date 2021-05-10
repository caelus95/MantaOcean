import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
DIR = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/yan_sun_eemds/'
w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/figs/eemds/'
w_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/figs/'

data_list = np.sort([file for file in os.listdir(DIR) if file.endswith(".npy")])

t = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m')#.to_period()

sig_list = []
n = 1
for i in data_list:
    tmp_data = np.load(DIR+i)
    sig_list.append(tmp_data[:,5])
    fig, axs = plt.subplots(tmp_data.shape[1],figsize=(9,13))
    axs[0].set_title('Yan Sun_'+f'{n:02d}',fontsize=28)
    for j in range(9):
        axs[j].plot(t, tmp_data[:,j])
    # plt.savefig(w_path1+f'{n:02d}'+'eemd')
    plt.show()
    n+=1
    if not i[15].isdigit():
        break


imf5_11 = pd.DataFrame(sig_list[10]-.1,columns=['data'])

date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')

imf5_11['date'] = date

# =============================================================================
# plotting
# =============================================================================


marker = 1

data = imf5_11.data - .1
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

plt.figure(1,figsize=(15,8),dpi=80)
ax = plt.gca()
plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
          fontweight='bold',fontsize='48')

for i in sig_list:
    plt.plot(imf5_11.date,i,color=[.5,.5,.5], linewidth=2,label=None)
plt.plot(imf5_11.date,np.mean(np.array(sig_list),axis=0),color=[.6,.1,.1],
         linewidth=3.5,linestyle='-.',label='Mean')

# Draw Plot
plt.plot(imf5_11.date,imf5_11.data, color='k', label='imf05(11th)')
if marker:
    plt.scatter(imf5_11.date[peak_locations], imf5_11.data[peak_locations],
                marker=mpl.markers.CARETUPBASE, color='darkred', s=100, label='Peaks')
    plt.scatter(imf5_11.date[trough_locations], imf5_11.data[trough_locations],
                marker=mpl.markers.CARETDOWNBASE, color='darkblue', s=100, label='Troughs')
    
    for ii in trough_locations:
        plt.text(imf5_11.date[ii], imf5_11.data[ii]-.02, imf5_11.date[ii], 
                 horizontalalignment='center', color='darkblue',fontsize=15)
    for jj in peak_locations:
        plt.text(imf5_11.date[jj], imf5_11.data[jj]+.008, imf5_11.date[jj], 
                 horizontalalignment='center', color='darkred',fontsize=15)
# Decoration
# plt.ylim(50,750)
xtick_location = imf5_11.index.tolist()[::12]
xtick_labels = imf5_11.date.tolist()[::12]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=45, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper left')
plt.grid(axis='y', alpha=.3)
# plt.savefig(w_path2+'imf5s')
plt.show()







# =============================================================================
# 
# =============================================================================




