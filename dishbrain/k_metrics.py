import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from collections import Counter



def plotMetrics(path):
    try:
        os.listdir(path)  
    except: 
        dir = []
    else:
        dir = os.listdir(path)
    distances = {}
    crashes = {}
    for k in dir:
        print(f'Mining for k={k}')
        seeds = os.listdir(path+'/'+k)
        if k not in distances: distances[k] = []
        if k not in crashes: crashes[k] = []
        for seed in seeds:    
            try: dates = os.listdir(path+'/'+k+'/'+seed) 
            except: break
            for date in dates:
                try:
                    data = pd.read_csv(path+'/'+k+'/'+seed+'/'+date+'/'+'dataFVW.csv')
                except:
                    break
                xy = [[data['x'][i],data['y'][i]] for i in range(len(data['x']))]

                distances[k].append(len(np.unique(xy, axis=0))/(np.pi*(40**2)-3*(np.pi*(3.5**2))))
                crashes[k].append(Counter(data['Crashes'])[1])
                print(f'Data saved for k={k}')
    
    try: labels
    except: pass
    else:
        print('Plotting and saving figures...')
        plt.figure()
        for k in labels:
            if k in distances:
                plt.bar(k,np.mean(distances[k]), color ='skyblue')
                plt.scatter([k]*len(distances[k]),distances[k], color = 'b')

        plt.xlabel('Stimulation')
        plt.ylabel('%')
        plt.title(f'Covered area')
        plt.savefig((path+'/coveredArea.png'))
        plt.clf()
        for k in labels:
            if k in crashes:
                plt.bar(k,np.mean(crashes[k]), color = 'skyblue')
                plt.scatter([k]*len(crashes[k]),crashes[k], color = 'b')
        plt.xlabel('Stimulation')
        plt.ylabel('N°')
        plt.title(f'Amount of crashes for k={k}')
        plt.savefig((path+'/crashes.png'))
            
        
# edsnn_path = 'results/sim_results/edsnn/IntegrationTime'
# labels = ['200.0ms','400.0ms','600.0ms','800.0ms','1000.0ms']
# plotMetrics(edsnn_path)

edsnn_path = 'results/sim_results/edsnn/k'
labels = ['0.1','0.5','1.0','1.0','IAF']
plotMetrics(edsnn_path)


