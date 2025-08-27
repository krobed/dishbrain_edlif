import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def moving_average(data):
    return np.sum(np.array(data).reshape(100,int(len(data)/100)),axis=1)

def plotMetrics(path):
    try:
        os.listdir(path)  
    except: 
        dir = []
    else:
        dir = os.listdir(path)
    test = 'ms'
    distances = {}
    crashes = {}
    for k in dir:
        seeds = os.listdir(path+'/'+k)
        for seed in seeds:    
            try: dates = os.listdir(path+'/'+k+'/'+seed) 
            except: break
            for date in dates:
                if date not in distances: distances[date] = []
                if date not in crashes: crashes[date] = []
                try:
                    data = pd.read_csv(path+'/'+k+'/'+seed+'/'+date+'/'+'dataFVW.csv')

                except:
                    break
                xy = [[data['x'][i],data['y'][i]] for i in range(len(data['x']))]

                distances[date].append(len(np.unique(xy, axis=0))/(np.pi*(40**2)-3*(np.pi*(3.5**2))))
                crashes[date].append(moving_average(data['Crashes']))
        if test == 'ms':
            try: dates
            except: pass
            else:
                plt.figure(figsize=(10,8))
                for date in labels:
                    plt.bar(date,np.mean(distances[date]), color ='skyblue')
                    plt.scatter([date]*len(distances[date]),distances[date], color = 'b')

                plt.xlabel('Stimulation')
                plt.ylabel('%')
                plt.title(f'Covered area')
                plt.savefig((path+'/'+k+'/coveredArea.png'))
                plt.clf()
                for date in labels:
                    plt.plot(np.mean(crashes[date], axis=0), color = 'red', label=date)
                plt.legend()
                plt.xlabel('Window')
                plt.ylabel('Mean crashes')
                plt.title(f'Mean crashes per window for k={k}')
                plt.savefig((path+'/'+k+'/crashes.png'))
            
        
# edsnn_path = 'results/sim_results/edsnn/IntegrationTime'
# labels = ['200.0ms','400.0ms','600.0ms','800.0ms','1000.0ms']
# plotMetrics(edsnn_path)

edsnn_path = 'results/sim_results/edsnn/Stimulation'
labels = ['RF0.0','RF2.0']#,'RF5.0','RF10.0','RF20.0','RF2.0 RAFR','RF5.0 RAFR','RF10.0 RAFR','RF20.0 RAFR']
plotMetrics(edsnn_path)


