# -*- coding: utf-8 -*-
#
# run_microcircuit.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""PyNEST Microcircuit: Run Simulation
-----------------------------------------

This is an example script for running the microcircuit model and generating
basic plots of the network activity.

"""
#%%
###############################################################################
# Import the necessary modules and start the time measurements.
import nest
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import argparse
from random import randint
import os
import pandas as pd
from environment import CustomCircleCarEnv
from utils.nest_utils import setup_nest
import os
import pickle

defaults = {'integration_time':1000.0, 'stimulation':2.0,'rfr':True, 'k':1,'sim_time':5.0}

#%%
args = argparse.ArgumentParser()
# args.add_argument('--backup', type=str, default="none") #not working
args.add_argument('--pre_model', type=str, default="edsnn") 
args.add_argument('--time', type=int, default=450000) #as wished
args.add_argument('--atp', type=int, default=100) #between 0 and 100
args.add_argument('--seed',type=int, default=66)
args.add_argument('--k',type=float,default=defaults['k']) #between 0 and 1
args.add_argument('--circle_size', type=float, default=0.4)
args.add_argument('--n_obstacles', type=int, default=3)
args.add_argument('--plots',type=bool,default=False)
args.add_argument('--sim_time', type=float, default=defaults['sim_time'])
args.add_argument('--integration_time', type=float, default=defaults['integration_time'])
args.add_argument('--stimulation', type=float, default = defaults['stimulation'])
args.add_argument('--rfr', type=bool, default = defaults['rfr'])
args.add_argument('--update', type=bool,default=False)

if __name__ == '__main__':
    args = args.parse_args()
    from assets.edsnn.stimulus_params import stim_dict
    from assets.edsnn.network_params import net_dict
    from assets.edsnn.sim_params import sim_dict
    from network import network_edsnn as network

    path = 'results/sim_results/edsnn/k/0.1/22/'

    try: os.listdir(path)
    except: raise FileNotFoundError
    else: 
        dir = path +os.listdir(path)[-1] + '/weights.csv'
        dataframe = pd.read_csv(dir)
        for w in dataframe['final_w']:
            if w<0:
                inh = w
                break
        index = list(dataframe['final_w']).index(inh)
        source_e = dataframe['source'][:index]
        source_e = np.array(source_e[source_e>=0], dtype=np.uint64)
        target_e = np.array(dataframe['target'][:index], dtype=np.uint64)
        w_e = np.array(dataframe['final_w'][:index])
        source_i = np.array(dataframe['source'][index:], dtype=np.uint64)
        target_i = np.array(dataframe['target'][index:], dtype=np.uint64)
        w_i = np.array(dataframe['final_w'][index:])
        n_order = np.max(source_e)//4
    if 'iaf' == args.pre_model:
        net_dict['neuron_params']['E']['gamma'] = 0
        net_dict['neuron_params']['I']['gamma'] = 0 
        net_dict['neuron_params']['eta'] = 0
        if args.plots:
            fig_sen, ax_sen = plt.subplots(figsize= (12,8))
            fig_neu, ax_neu = plt.subplots(figsize= (12,8))
        neuron_model = 'iaf'
        # print('Configuring for IAF model')
    else:
        if args.plots:
            fig_sen, ax_sen = plt.subplots(2, 1, figsize= (12,8))
            fig_neu, ax_neu = plt.subplots(2, 1, figsize= (12,8))
        neuron_model = 'edsnn'
        # print('Configuring for EDSNN model')
    sim_dict['t_sim'] = args.sim_time
    if not args.update: net_dict['lambda'] = 0
    
    sim_dict['rng_seed']= int(args.seed)

    seed=sim_dict['rng_seed']
    net_dict['full_num_neurons'] = [n_order*4,n_order]
    net_dict['neuron_params']['E']['ATP'] = args.atp
    net_dict['neuron_params']['I']['ATP'] = args.atp
    net_dict['neuron_params']['E']['K_ATP'] = args.k
    net_dict['neuron_params']['I']['K_ATP'] = args.k
    k= net_dict['neuron_params']['E']['K_ATP']
    time_start = time.time()
    sim_dict['integration_time'] = args.integration_time
    # Make path to save results

    t=datetime.now().strftime('%Y%m%d%H%M%S')
    path= fr"{os.path.dirname(os.path.abspath(__file__))}/results/testing/{neuron_model}/{seed}/{k}/{t}"
    if not os.path.exists(path):
        os.makedirs(path)

    ###############################################################################
    # Initialize the network with simulation, network and stimulation parameters,
    # then create and connect all nodes, and finally simulate.
    # The times for a presimulation and the main simulation are taken
    # independently. A presimulation is useful because the spike activity typically
    # exhibits a startup transient. In benchmark simulations, this transient should
    # be excluded from a time measurement of the state propagation phase. Besides,
    # statistical measures of the spike activity should only be computed after the
    # transient has passed.

    # Nest Setup
    setup_nest(sim_dict)

    if net_dict["synapse_model"] is not None:
        wr_e = nest.Create("weight_recorder")
        nest.CopyModel(
            net_dict["synapse_model_lib"], 
            'edstdp_e',
            {
                "weight_recorder": wr_e,
            }
        )
        wr_i = nest.Create("weight_recorder")
        nest.CopyModel(
            'static_synapse',
            'static_i',
            {
                "weight_recorder": wr_i,
            }
        )
    
    net_dict['n_sensors'] = 20
    # Network
    net = network.Network(sim_dict, net_dict, stim_dict)

    time_network = time.time()
    # Create
    net.create()
    time_create = time.time()
    # Connect
    syn_dict_e = {
            'synapse_model': 'edstdp_e',
            'alpha': net_dict['alpha'],
            'mu_plus': net_dict['mu_plus'],
            'mu_minus': net_dict['mu_minus'],
            'lambda': net_dict['lambda'],
            'eta': net_dict['eta'],
            'Wmax': 100
        }
    syn_dict_e['w'] = w_e
    nest.Connect(source_e,target_e,conn_spec={'rule':'one_to_one'}, syn_spec=syn_dict_e)
    nest.Connect(source_i,target_i,conn_spec={'rule':'one_to_one'}, syn_spec={'synapse_model':'static_i', 'weight':w_i})
    # Generate and connect noise signal
    n = sum(net.num_neurons)-net.n_sensor_neuron
    noisep = nest.Create('poisson_generator',int(n))
    noisep.rate = 10
    nest.Connect(noisep, net.pops[0][net.n_sensor_neuron:]+net.pops[1], conn_spec={'rule':'one_to_one'},syn_spec={"weight": 400})

    
    net.connect_recording()
    time_connect = time.time()

    # Nest Prepare and Cleanup
    nest.set_verbosity("M_ERROR")
    nest.Prepare()
    nest.Cleanup()

    # Reset environment
    # obs = env.reset()

    # Set sensors
    if net.n_sensors == 1:
        sensors_angle = [0]
    else:
        sensors_angle = np.linspace(-np.pi/2,np.pi/2,net.n_sensors+1)[:-1]

    # Environment
    env = CustomCircleCarEnv(
        net.n_sensors,
        sensors_angle,
        args.sim_time/1000,
        radious=args.circle_size,
        n_obstacles=args.n_obstacles
    )
    # Simulation parameters
    simulation_time = sim_dict['t_sim']
    epochs = int(args.time//simulation_time)
    firing_rate = [0]*net.n_sensors
    if args.plots:
        distance, ax_d = plt.subplots()
        crashes, ax_c = plt.subplots()
        fig_out, ax_out = plt.subplots(3, 1, figsize= (8,10))
        axis = [ax_c,ax_d,ax_neu,ax_out,ax_sen]
    # print(nest.GetDefaults(net_dict["neuron_model"])['recordables'])

    data = {'fr_Input':[], 'fr_Left':[], 'fr_Right':[]} # Init dictionary for firing rates
    data['v_Left']=[] # velocities
    data['v_Right']=[] 
    spikes={}
    spikes['sensor_S']=[] # spike times
    spikes['neuron_S']=[] 
    init_weights ={}
    final_weights ={}
    
    epoch_saved = 0 #Init epoch saved
    sim_time_saved = 0 #Init simulation time saved
    data['Distance'] = []
    data['Crashes'] = []
    data['x']=[]
    data['y']=[]
    robot_A = np.pi *(3.5**2)
    area_covered_init = 1/(np.pi*(40**2) - len(env.obstacles)*robot_A)
    V_values = {'times':[]}
    ATP_values = {'times':[]}
    for i in range(net.n_sensor_neuron):
        V_values[f'Sensor {i}'] = []
        ATP_values[f'Sensor {i}'] = []
    for i in range(net.n_recorders):
        V_values[f'Neuron {i}'] = []
        ATP_values[f'Neuron {i}'] = []

    weights = {'source':[],'target':[],'init_w':[]}
    
    # firing_rate = [5]*net.n_sensors  # test with some firing rate
    # Get initial weights
    if net_dict["synapse_model"] is not None:
        syn = nest.GetConnections(
            synapse_model='edstdp_e')
        syn_i = nest.GetConnections(
            synapse_model='static_i')
        weights['source']= [c.source for c in syn]
        weights['target']= [c.target for c in syn]
        weights['init_w'] = list(nest.GetStatus(syn, 'w'))
        weights['source'].extend([c.source for c in syn_i])
        weights['target'].extend([c.target for c in syn_i])
        weights['init_w'].extend(nest.GetStatus(syn_i, 'weight'))

    # Start simulations:

    last_firing_rates = firing_rate
    total_spike_times = [np.array([]) for i in range(net.n_sensors)]
    t = 0
    for epoch in range(1,epochs+1): 
        
        if epochs==epoch:
            syn = nest.GetConnections(synapse_model='edstdp_e')
            syn_i = nest.GetConnections(synapse_model='static_i')

        data['x'].append(np.around(env.trajectory[-1][0],2))
        data['y'].append(np.around(env.trajectory[-1][1],2))

        # print()
        # print(f"Epoch {epoch}, firing_rate: {sum(firing_rate)}")
        # print(f"Covered Area: {data['Distance'][-1]}")
        # print(f"Number of colisons: {sum(data['Crashes'])}")
        # print("actual_time:", net.actual_simulation_time)
    
        # Generate spike times
       
        spike_times = [] 
        
        for i in range(len(firing_rate)):
            if last_firing_rates[i]!=firing_rate[i]:
                if firing_rate[i]!=0:
                    if len(total_spike_times[i])!=0:
                        n_spikes = (total_spike_times[i][-1] - total_spike_times[i][0])*firing_rate[i]/1000
                        times= np.around(np.linspace(total_spike_times[i][0], total_spike_times[i][-1], round(n_spikes+1), endpoint=False))  
                    else:  
                        n_spikes = (args.time - net.actual_simulation_time)*firing_rate[i]/1000
                        times= np.around(np.linspace(net.actual_simulation_time+2, args.time, round(n_spikes+1), endpoint=False))
                    total_spike_times[i] = times
                else:
                    total_spike_times[i] = np.array([])
                last_firing_rates[i] = firing_rate[i]
        

        t += simulation_time
        for ts in total_spike_times:
            pts = ts[ts<=t]
            pts = pts[pts>=net.actual_simulation_time]            
            spike_times.append(pts)

        # External stimulation
        net.external_stimulation(
            spike_times=spike_times)

        

        # Simulate
        net.simulate(
            t_sim=simulation_time)
        
        # Action
        action = net.decode_actions()
   
        # Step
        obs, firing_rate, done, info = env.step(action)
        firing_rate= np.array(firing_rate)
        

        if done:
            # print("-----------------------------------------------------------------------------------------------------------------------------CRASH")
            data['Crashes'].append(1)
            obs = env.reset()
            if args.rfr:
                firing_rate=[2]*net.n_sensors
        else:
            data['Crashes'].append(0)
        # Plots
        data['fr_Input'].append(sum(firing_rate)/net.n_sensors)
        data['fr_Left'].append(round(net.left_fr,2))
        data['fr_Right'].append(round(net.right_fr,2))
        data['v_Left'].append(round(env.velocity_left_wheel,2))
        data['v_Right'].append(round(env.velocity_right_wheel,2))
        
        
        # Render the environment
        env.render()

        """
        Plotter configuration
        """

        if epoch%60000==0 or epoch==epochs or epoch==epochs*3/2: # Every 50 epochs
            # print('Plotting and saving data')
            # Updated weights
            if net_dict["synapse_model"] is not None and epoch == epochs and args.update:
                for c in syn:
                    final_weights[(c.source, c.target)] = c.w
                for c in syn_i:
                    final_weights[(c.source, c.target)] = c.weight
                weights['final_w'] = []
                for i in range(len(weights['source'])):
                    weights["final_w"].append(final_weights[(weights['source'][i],weights['target'][i])])
                
            start_time = epoch_saved * simulation_time
            end_time = net.actual_simulation_time
            if args.plots:
                for ax in axis:
                    try:
                        for a in ax:
                            a.clear()
                    except:
                        ax.clear()
                c = ["C0","C1"]
                vl=["Sensors","Neurons"]

                if neuron_model =="edsnn":
                    sen_vol = ax_sen[0]
                    neu_vol = ax_neu[0]
                    sen_atp = ax_sen[1]
                    neu_atp = ax_neu[1]
                else:
                    sen_vol = ax_sen
                    neu_vol = ax_neu
            """
            Plotting voltages
            """
            # print('Plotting voltages')
            voltages_sen = nest.GetStatus(net.voltmeters[:net.n_sensor_neuron],'events')
            volt = {'times': voltages_sen[0]['times'][voltages_sen[0]['times']>=start_time]}

            volt['Sensors'] = [voltages_sen[i]['V_m'][-len(volt['times']):] for i in range(net.n_sensor_neuron)]
            voltages_neu = nest.GetStatus(net.voltmeters[-net.n_recorders:],'events')
            volt['Neurons'] = [voltages_neu[i]['V_m'][-len(volt['times']):] for i in range(net.n_recorders)]
            V_values['times']= voltages_sen[0]['times']
            if args.plots:
                for i in range(len(volt['Sensors'])):
                    sen_vol.plot(volt['times'], volt['Sensors'][i],c[0])
                    V_values[f'Sensor {i}']=np.around(voltages_sen[i]['V_m'],2)
                    
                for i in range(len(volt['Neurons'])):
                    neu_vol.plot(volt['times'],volt['Neurons'][i],c[1])
                    V_values[f'Neuron {i}']=np.around(voltages_neu[i]['V_m'],2)
                    
                sen_vol.set_title("Voltages of neurons connected to sensors")
                neu_vol.set_title("Voltages of measured neurons")
                sen_vol.set_ylabel("V_m [mV]")
                neu_vol.set_ylabel("V_m [mV]")
                neu_vol.grid()
                sen_vol.grid()
            else:
                for i in range(len(volt['Sensors'])):
                    V_values[f'Sensor {i}']=np.around(voltages_sen[i]['V_m'],2)
                    
                for i in range(len(volt['Neurons'])):
                    V_values[f'Neuron {i}']=np.around(voltages_neu[i]['V_m'],2)

            """
            Plotting spikes
            """
            # print('Plotting spikes')
            ss_times = nest.GetStatus(net.spike_recorders[:net.n_sensor_neuron], 'events')
            ns_times = nest.GetStatus(net.spike_recorders[net.n_sensor_neuron:], 'events')
            s_times = {'Sensor Spikes': [ss['times'][ss['times']>=start_time] for ss in ss_times]}
            s_times['Neuron Spikes'] = [ns['times'][ns['times']>=start_time] for ns in ns_times]
            # print(len(s_times['Neuron Spikes']))
            if args.plots:
                for i in range(len(s_times['Sensor Spikes'])):
                    if i==0:
                        sen_vol.plot(s_times['Sensor Spikes'][i], -67 * np.ones(len(s_times['Sensor Spikes'][i])), "ok", label = 'spikes')
                    else:
                        sen_vol.plot(s_times['Sensor Spikes'][i], -67 * np.ones(len(s_times['Sensor Spikes'][i])), "ok")
                for i in range(len(s_times['Neuron Spikes'])):
                    if i==0:
                        neu_vol.plot(s_times['Neuron Spikes'][i], -67 * np.ones(len(s_times['Neuron Spikes'][i])), "ok", label = 'spikes')
                    else:
                        neu_vol.plot(s_times['Neuron Spikes'][i], -67 * np.ones(len(s_times['Neuron Spikes'][i])), "ok")
                sen_vol.legend()
                neu_vol.legend()

            """
            Plotting ATP (just for EDLIF)
            """
            # print('Plotting ATP')
            if neuron_model == 'edsnn':
                ATP_sen = nest.GetStatus(net.atpmeters[:net.n_sensor_neuron], 'events')
                ATP = {'times':ATP_sen[0]['times'][ATP_sen[0]['times']>=start_time]}
                ATP['Sensors'] = [ATP_sen[i]['ATP'][-len(ATP['times']):] for i in range(net.n_sensor_neuron)]
                ATP_neu = nest.GetStatus(net.atpmeters[net.n_sensor_neuron:], 'events')
                ATP['Neurons'] = [ATP_neu[i]['ATP'][-len(ATP['times']):] for i in range(net.n_recorders)]
                ATP_values['times']= ATP_sen[0]['times']
                if args.plots:
                    for i in range(len(ATP['Sensors'])):
                        sen_atp.plot(ATP['times'], ATP['Sensors'][i],c[0])
                        ATP_values[f'Sensor {i}']=np.around(ATP_sen[i]['ATP'],2)
                    for i in range(len(ATP['Neurons'])):
                        neu_atp.plot(ATP['times'],ATP['Neurons'][i],c[1])
                        ATP_values[f'Neuron {i}']=np.around(ATP_neu[i]['ATP'],2)
                    sen_atp.set_title("ATP for neurons connected to sensors")
                    neu_atp.set_title("ATP for measured neurons")
                    sen_atp.set_xlabel("time [ms]")
                    neu_atp.set_xlabel("time [ms]")
                    sen_atp.set_ylabel("ATP [%]")
                    neu_atp.set_ylabel("ATP [%]")
                    neu_atp.grid()
                    sen_atp.grid()  
                else:
                    for i in range(len(ATP['Sensors'])):
                        ATP_values[f'Sensor {i}']=np.around(ATP_sen[i]['ATP'],2)
                    for i in range(len(ATP['Neurons'])):
                        ATP_values[f'Neuron {i}']=np.around(ATP_neu[i]['ATP'],2)
            elif args.plots:
                ax_sen.set_xlabel("time [ms]")
                ax_neu.set_xlabel("time [ms]")

            if args.plots:
                """
                Plotting firing rates
                """
                # print('Plotting firing rates')
                ax_out[0].plot(data['fr_Input'][epoch_saved:], "k", label = "Input")
                ax_out[0].plot(data['fr_Left'][epoch_saved:], "C0", label="Left")
                ax_out[0].plot(data['fr_Right'][epoch_saved:], "C1", label="Right")
                ax_out[0].set_title("Firing rates")
                ax_out[0].legend()
                ax_out[0].grid()
                ax_out[0].set_ylabel(r'$\upsilon$')
                """
                Plotting velocities
                """
                # print('Plotting velocities')
                ax_out[1].plot(data['v_Left'][epoch_saved:], "C0", label="Left")
                ax_out[1].plot(data['v_Right'][epoch_saved:], "C1", label="Right")
                ax_out[1].set_title("Velocities")
                ax_out[1].legend()
                ax_out[1].grid()
                ax_out[1].set_ylabel('rads/s')

                """
                Plotting weights
                """
                # print('Plotting weights')

                # for i in range(len(weights)):
                #     ax_out[2].scatter(range(epoch_saved,epoch),weights[f'Neuron {i}'][-epoch+epoch_saved:])
                # ax_out[2].set_title("Weights")
                # ax_out[2].set_xlabel("Epoch N°")
                # ax_out[2].set_ylabel('w')

                """
                Plotting covered area and crash count
                """
                # print('Plotting covered area and crash count')
                ax_d.plot(np.linspace(0,net.actual_simulation_time/1000, len(data['Distance'])),data['Distance'])
                ax_d.fill_between(np.linspace(0,net.actual_simulation_time/1000, len(data['Distance'])),data['Distance'])
                ax_d.set_title("Percentage of area covered")
                ax_d.set_ylabel("%")
                ax_d.set_xlabel("Epoch N°")
                ax_c.plot(np.linspace(0,net.actual_simulation_time/1000, len(data['Crashes'])), data['Crashes'])
                ax_c.fill_between(np.linspace(0,net.actual_simulation_time/1000, len(data['Crashes'])), data['Crashes'])
                ax_c.set_title("Instant of crashes")
                ax_c.set_xlabel("Epoch N°")
                ax_c.set_ylabel("N°")
                fig_sen.tight_layout()
                fig_neu.tight_layout()
                fig_out.tight_layout()
                distance.tight_layout()
                crashes.tight_layout()

                """
                Save figures
                """
                # print('Saving figures')
                fig_sen.savefig(fr'{path}/DataSensors {epoch}.png')
                fig_neu.savefig(fr'{path}/DataNeurons {epoch}.png')
                fig_out.savefig(fr'{path}/DataFiringrateVelocityWeight {epoch}.png')
                distance.savefig(fr'{path}/AreaCovered.png')
                crashes.savefig(fr'{path}/CrashCount.png')
            
            env.fig.savefig(fr'{path}/Trayectory.png')
            """
            Save data
            """
            # print('Saving data')
            # print('Saving FVAC')
            pd.DataFrame(data).to_csv(fr'{path}/dataFVW.csv', index= True)
            # print('Saving voltages')
            pd.DataFrame(V_values).to_csv(fr'{path}/Voltages.csv', index= False)
            # print('Saving weights')
            pd.DataFrame(weights).to_csv(fr'{path}/weights.csv',index=False)
            if neuron_model == 'edsnn':
                # print('Saving ATP')
                pd.DataFrame(ATP_values).to_csv(fr'{path}/ATP.csv', index= False)
            epoch_saved = epoch-1
            sim_time_saved = int(net.actual_simulation_time-1)
    
    env.close()

