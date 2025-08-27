import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from collections import Counter

edsnn_path = 'results/sim_results/edsnn'
iaf_path = 'results/sim_results/iaf'

try:
    os.listdir(edsnn_path)  
except: 
    edsnn_directories = []
else:
    edsnn_directories = os.listdir(edsnn_path)


try:
    os.listdir(iaf_path)    
except: 
    iaf_directories = []
else:
    iaf_directories = os.listdir(iaf_path)


for seed in edsnn_directories:
    ks = os.listdir(edsnn_path+'/'+seed)
    for k in ks:
        dates = os.listdir(edsnn_path+'/'+seed+'/'+k)
        for date in dates:
            try:
                weights = pd.read_csv(edsnn_path+'/'+seed+'/'+k+'/'+date+'/'+'weights.csv')
            except:
                break
            init_w = list(weights['init_w'])
            final_w = list(weights['final_w'])
            source = list(weights['source'])
            n=0
            while n<len(source):
                if float(source[n])<=20:
                    final_w.pop(n)
                    init_w.pop(n)
                    source.pop(n)
                else:
                    n+=1

            count = Counter(final_w)
            keys = list(count.keys())
            values = list(count.values())
            
            n=0
            inh_k = []
            inh_v = []
            while n < len(keys):
                if keys[n]<0:
                    inh_k.append(keys.pop(n))
                    inh_v.append(values.pop(n))
                else:
                    n+=1
                

            plt.grid()
            plt.bar(keys,values)
            plt.title(f'Final weights EDSNN k={k}')
            plt.xlabel('Weights')
            plt.ylabel('Frecuency')
            plt.savefig(edsnn_path+'/'+seed+'/'+k+'/'+date+'/'+'final_w.png')
            plt.clf()
            count = Counter(init_w)
            keys = list(count.keys())
            values = list(count.values())
            source = list(weights['source'])
            n=0
            inh_k = []
            inh_v = []
            while n < len(keys):
                if keys[n]<0:
                    inh_k.append(keys.pop(n))
                    inh_v.append(values.pop(n))
                else:
                    n+=1
                
            plt.grid()
            plt.bar(keys,values)
            plt.title(f'Initial weights EDSNN k={k}')
            plt.xlabel('Weights')
            plt.ylabel('Frecuency')
            plt.savefig(edsnn_path+'/'+seed+'/'+k+'/'+date+'/'+'init_w.png')
            plt.clf()

for seed in iaf_directories:
    ks = os.listdir(iaf_path+'/'+seed)
    for k in ks:
        dates = os.listdir(iaf_path+'/'+seed+'/'+k)
        for date in dates:
            try:
                weights = pd.read_csv(edsnn_path+'/'+seed+'/'+k+'/'+date+'/'+'weights.csv')
            except:
                break
            init_w = list(weights['init_w'])
            final_w = list(weights['final_w'])
            source = list(weights['source'])
            n=0
            while n<len(source):
                if float(source[n])<=62:
                    final_w.pop(n)
                    init_w.pop(n)
                    source.pop(n)
                else:
                    n+=1

            count = Counter(final_w)
            keys = list(count.keys())
            values = list(count.values())
            n=0
            inh_k = []
            inh_v = []
            while n < len(keys):
                if keys[n]<0:
                    inh_k.append(keys.pop(n))
                    inh_v.append(values.pop(n))
                else:
                    n+=1
                
            plt.grid()
            plt.bar(keys,values)
            plt.title('Final weights IAF')
            plt.xlabel('Weights')
            plt.ylabel('Frecuency')
            plt.savefig(iaf_path+'/'+seed+'/'+k+'/'+date+'/'+'final_w.png')
            plt.clf()
            count = Counter(init_w)
            keys = list(count.keys())
            values = list(count.values())
            source = list(weights['source'])
            n=0
            inh_k = []
            inh_v = []
            while n < len(keys):
                if keys[n]<0:
                    inh_k.append(keys.pop(n))
                    inh_v.append(values.pop(n))
                else:
                    n+=1
                
            plt.grid()
            plt.bar(keys,values)
            plt.title('Initial weights IAF')
            plt.xlabel('Weights')
            plt.ylabel('Frecuency')
            plt.savefig(iaf_path+'/'+seed+'/'+k+'/'+date+'/'+'init_w.png')
            plt.clf()