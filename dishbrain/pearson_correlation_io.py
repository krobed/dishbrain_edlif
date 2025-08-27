import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import os
import matplotlib.pyplot as plt

def pearson_corr(path):
    try:
        os.listdir(path)  
    except: 
        dir = []
    else:
        dir = os.listdir(path)
    firing_rates = {}
    for k in dir:
        seeds = os.listdir(path+'/'+k)
        for seed in seeds:    
            try: dates = os.listdir(path+'/'+k+'/'+seed) 
            except: break
            for date in dates:
                if date == 'RF0.0': continue
                if date not in firing_rates: firing_rates[date] = {'InputL':[], 'InputR':[],'Left':[], 'Right':[]}
                try:
                    data = pd.read_csv(path+'/'+k+'/'+seed+'/'+date+'/'+'dataFVW.csv')
                except:
                    break

                # Get the columns
                il_rates = data['frI_Left']
                ir_rates = data['frI_Right']
                ol_rates = data['frO_Left']
                or_rates = data['frO_Right']
                # Compute Pearson correlation
                firing_rates[date]['ILeft'].extend(il_rates)
                firing_rates[date]['IRight'].extend(ir_rates)
                firing_rates[date]['OLeft'].extend(ol_rates)
                firing_rates[date]['ORight'].extend(or_rates)
    plt.scatter(firing_rates[date]['Input'], firing_rates[date]['Left'])
    plt.scatter(firing_rates[date]['Input'], firing_rates[date]['Right'])
    plt.show()

edsnn_path = 'results/sim_results/2000/Stimulation'
labels = ['RF0.0','RF2.0', ]#,'RF5.0','RF10.0','RF20.0','RF2.0 RAFR','RF5.0 RAFR','RF10.0 RAFR','RF20.0 RAFR']
pearson_corr(edsnn_path)