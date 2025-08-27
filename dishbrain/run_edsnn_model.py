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
import psth
from alive_progress import alive_bar

import gc

def print_diagnostics(recorders=None, label=""):
    print("\n=== Diagnostics", label, "===")
    # Neurons
    neurons = nest.GetNodes()[0]
    print("Neurons:", len(neurons))

    # Connections
    conns = nest.GetConnections()
    print("Connections:", len(conns))

    # Recorders
    if recorders is not None:
        for r in recorders:
            st = nest.GetStatus(r)[0]
            n_events = st.get("n_events", "N/A")
            print(f"Recorder {r} -> n_events={n_events}, record_to={st.get('record_to')}")

    # Memory from kernel (per MPI rank)
    mem = nest.ll_api.sli_func("memory_thisjob")  # kB
    print("Kernel memory usage (per rank): %.2f MB" % (mem / 1024.0))

    gc.collect()
    print("================================\n")

defaults = {'integration_time': 500.0, 'stimulation':40.0,'rfr':False, 'k':1,'sim_time':50.0}

args = argparse.ArgumentParser()
# args.add_argument('--backup', type=str, default="none") #not working
args.add_argument('--neuron_order', type=int, default=200) #as wished / dictates number of inhibitory neurons. Ratio of exc:inh = 4:1
args.add_argument('--time', type=int, default= 300000) #as wished
args.add_argument('--atp', type=int, default=100) #between 0 and 100
args.add_argument('--neuron_model', type=str, default='edsnn') #'edsnn' or 'iaf'
args.add_argument('--seed',type=int, default=66)
args.add_argument('--k',type=float,default=defaults['k']) #between 0 and 1
args.add_argument('--circle_size', type=float, default=0.4)
args.add_argument('--n_obstacles', type=int, default=3)
args.add_argument('--plots',type=bool,default=False)
args.add_argument('--test', type=bool,default=False)
args.add_argument('--sim_time', type=float, default=defaults['sim_time'])
args.add_argument('--integration_time', type=float, default=defaults['integration_time'])
args.add_argument('--stimulation', type=float, default = defaults['stimulation'])
args.add_argument('--rfr', type=bool, default = defaults['rfr'])



if __name__ == '__main__':
    args = args.parse_args()
    from assets.edsnn.stimulus_params import stim_dict
    from assets.edsnn.network_params import net_dict
    from assets.edsnn.sim_params import sim_dict
    from network import network_edsnn as network
    if args.neuron_model == 'iaf':
        net_dict['neuron_params']['E']['gamma'] = 0
        net_dict['neuron_params']['I']['gamma'] = 0
        net_dict['neuron_params']['eta'] = 0
        if args.plots:
            fig, ax = plt.subplots(figsize= (12,8))
        neuron_model = 'iaf'
        k='LIF'
        # print('Configuring for IAF model')
    else:
        if args.plots:
            fig, ax = plt.subplots(2, 1, figsize= (12,8))
        neuron_model = 'edsnn'
        # print('Configuring for EDSNN model')
    sim_dict['t_sim'] = args.sim_time
    sim_dict['rng_seed']= int(args.seed)

    seed=sim_dict['rng_seed']
    net_dict['full_num_neurons'] = [round(args.neuron_order*4),round(args.neuron_order)]
    net_dict['neuron_params']['E']['ATP'] = args.atp
    net_dict['neuron_params']['I']['ATP'] = args.atp
    net_dict['neuron_params']['E']['K_ATP'] = args.k
    net_dict['neuron_params']['I']['K_ATP'] = args.k
    k= net_dict['neuron_params']['E']['K_ATP']
    time_start = time.time()
    sim_dict['integration_time'] = args.integration_time
    # Make path to save results
    if defaults['integration_time']!=args.integration_time:
        test = 'IntegrationTime'
    elif defaults['stimulation']!=args.stimulation or defaults['rfr']!= args.rfr:
        test = 'Stimulation'
    elif defaults['sim_time']!=args.sim_time:
        test = 'NavigationTime'
    else:
        test='k'
    
    
    if test == 'IntegrationTime':
        value = str(args.integration_time)+'ms'
    elif test == 'k':
        value = datetime.now().strftime('%Y%m%d%H%M%S')
    elif test=='NavigationTime':
        value = str(args.sim_time)+'ms'
    elif test == 'Stimulation':
        value = 'RF'+str(args.stimulation)
        if args.rfr:
            value += ' RAFR'
    path= fr"{os.path.dirname(os.path.abspath(__file__))}/results/sim_results/{sum(net_dict['full_num_neurons'])}/{test}/{k}/{seed}/{value}"
    sim_dict['path'] =path
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

    net_dict['n_sensors'] = 6
    # Network
    net = network.Network(sim_dict, net_dict, stim_dict)
    time_network = time.time()
    # Create
    net.create()
    time_create = time.time()
    # Connect
    net.connect()
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
        # print(sensors_angle)

    # Environment
    env = CustomCircleCarEnv(
        net.n_sensors,
        sensors_angle,
        args.sim_time/1000,
        stimulation = args.stimulation,
        radious=args.circle_size,
        n_obstacles=args.n_obstacles
    )
    # Simulation parameters
    simulation_time = sim_dict['t_sim']
    epochs = int(args.time//simulation_time)
    firing_rate = [0]*net.n_sensors
    if args.plots:
        crashes, ax_c = plt.subplots()
        fig_out, ax_out = plt.subplots(2, 1, figsize= (8,8))
        axis = [ax_c,ax,ax_out]
    # print(nest.GetDefaults(net_dict["neuron_model"])['recordables'])

    data = {'frI_Left':[], 'frI_Right':[], 'frO_Left':[], 'frO_Right':[]} # Init dictionary for firing rates
    data['v_Left']=[] # velocities
    data['v_Right']=[]
  
    epoch_saved = 0 #Init epoch saved
    sim_time_saved = 0 #Init simulation time saved
    data['x'] = []
    data['y'] = []
    data['Crashes'] = []
    robot_A = np.pi *(3.5**2)
    area_covered_init = 1/(np.pi*(40**2) - len(env.obstacles)*robot_A)
        

    # Start simulations:
    if args.test: total_epochs =  epochs+epochs//2 +1
    else: total_epochs = epochs+1
    last_firing_rates = [0]*net.n_sensors
    total_stimes_right = [np.array([]) for i in range(round(net.n_sensors/2))]
    total_stimes_left = [np.array([]) for i in range(round(net.n_sensors/2))]
    
    print('Starting Simulation')
    # Pre-stimulations:
    random_electrodes = np.random.randint(0,len(net.electrodes),len(net.electrodes)//2) #Take 5 random electrodes to stimulate
    print(f'Chosen electrodes: {random_electrodes}')
    phase_width_ms = 0.5   # each phase
    pamplitude_pA = 1000
    namplitude_pA = -500
    n_pulses=50
    for electrode in random_electrodes:
        print(f'Stimulating electrode {electrode}')
        # Times at which spikes occur (start at 0 ms)
        if net.actual_simulation_time==0.0: t=1.0+net.actual_simulation_time
        else: t=net.actual_simulation_time
        p_times = np.arange(t, n_pulses * phase_width_ms*2 + t, 1.0)
        n_times = np.arange(t + phase_width_ms, n_pulses * phase_width_ms*2 + t+phase_width_ms, 1.0)
        p_stim = nest.Create("spike_generator", params={"spike_times": p_times.tolist()})
        n_stim = nest.Create("spike_generator", params={"spike_times": n_times.tolist()})
        # Connect with a weight that corresponds to desired current amplitude
        nest.Connect(p_stim, net.electrodes[electrode], syn_spec={"weight": pamplitude_pA})  # weight in pA
        nest.Connect(n_stim, net.electrodes[electrode], syn_spec={"weight": namplitude_pA})  # weight in pA
        net.simulate(2*phase_width_ms*n_pulses+400)
        rec = [net.spike_recorders]
        if 'atpmeter' in sim_dict['rec_dev']: rec.append(net.atpmeters)
        if 'voltmeter' in sim_dict['rec_dev']: rec.append(net.voltmeters)
        print_diagnostics(recorders=rec, label=f"after {net.actual_simulation_time} ms")
    print('Calculating correlations and PSTH')
    net.get_input_output()
    t_init = net.actual_simulation_time
    print(f'Actual simulation time: {t_init}')
    # Final simulations
    # with open(fr'{path}/dataFVW.csv', 'a') as f:
    #     f.write(f'# Electrodes used {net.best_pair}.\n')
    pd.DataFrame(data).to_csv(fr'{path}/dataFVW.csv', index= False, mode='a')
    with open(f"{path}/Vm_all.dat", "a") as f:
        f.write("# sender\ttime_ms\tV_m\n")
    with open(f"{path}/ATP_all.dat", "a") as f:
        f.write("# sender\ttime_ms\tATP\n")
    epochs_with_stimulation=0
    for epoch in range(1,total_epochs):
        if epochs==epoch and args.test:
            env.state=np.array([None,None,None])
            env.crash=[]
            env.reset()
            syn = nest.GetConnections(synapse_model='edstdp_e')
            syn_i = nest.GetConnections(synapse_model='static_i')
            nest.SetStatus(syn, {'lambda':0.0})
            # New environment
            # Environment
            env = CustomCircleCarEnv(
                net.n_sensors,
                sensors_angle,
                args.sim_time/1000,
                stimulation = args.stimulation,
                radious=args.circle_size*2,
                n_obstacles=args.n_obstacles*3
            )
        elif epochs==epoch:
            syn = nest.GetConnections(synapse_model='edstdp_e')
            syn_i = nest.GetConnections(synapse_model='static_i')

        
        data['x'].append(np.around(env.trajectory[-1][0],2))
        data['y'].append(np.around(env.trajectory[-1][1],2))

        # print()
        # print(f"Epoch {epoch}, firing_rate: {sum(firing_rate)}")
        # print(f"Covered Area: {data['Trajectory'][-1]}")
        # print(f"Number of colisons: {sum(data['Crashes'])}")
        # print("actual_time:", net.actual_simulation_time)

        # Generate spike times

        spike_times = []
        last_fr_right = last_firing_rates[:round(len(firing_rate)/2)]
        fr_right = firing_rate[:round(len(firing_rate)/2)]
        last_fr_left = last_firing_rates[round(len(firing_rate)/2):]
        fr_left = firing_rate[round(len(firing_rate)/2):]
        
        # Get real spike times from new firing rates:
        if sum(last_fr_left) != sum(fr_left) :
            if sum(total_stimes_left)!=0:
                n_spikes = (total_stimes_left[-1] - total_stimes_left[0])*max(fr_left)/1000
                times= np.around(np.linspace(total_stimes_left[0], total_stimes_left[-1], round(n_spikes+1), endpoint=False))  
            else:  
                n_spikes = (args.time+ t_init - net.actual_simulation_time)*max(fr_left)/1000
                times= np.around(np.linspace(net.actual_simulation_time +1.0 if net.actual_simulation_time==0 else net.actual_simulation_time, args.time + t_init, round(n_spikes), endpoint=False))
            total_stimes_left = times
        elif sum(fr_left) ==0 :
            total_stimes_left = np.array([])
        last_fr_left = fr_left
        if sum(last_fr_right) != sum(fr_right):
            if sum(total_stimes_right)!=0:
                n_spikes = (total_stimes_right[-1] - total_stimes_right[0])*max(fr_right)/1000
                times= np.around(np.linspace(total_stimes_right[0], total_stimes_right[-1], round(n_spikes+1), endpoint=False))  
            else:  
                n_spikes = (args.time+t_init - net.actual_simulation_time)*max(fr_right)/1000
                times= np.around(np.linspace(net.actual_simulation_time +1.0 if net.actual_simulation_time==0 else net.actual_simulation_time, args.time + t_init, round(n_spikes), endpoint=False))
            total_stimes_right = times
        elif sum(fr_right) ==0 :
            total_stimes_right = np.array([])
        last_fr_right = fr_right

        # Get only useful spike times
        t = net.actual_simulation_time + simulation_time
        left = total_stimes_left[total_stimes_left<=t]
        total_stimes_left = total_stimes_left[total_stimes_left>t]           
        right = total_stimes_right[total_stimes_right<=t]
        total_stimes_right = total_stimes_right[total_stimes_right>t]
        spike_times.append(left)
        spike_times.append(right)

        # External stimulation
        net.external_stimulation(
            spike_times)



        # Simulate
        net.simulate(
            t_sim=simulation_time)

        # Save Voltages
        if 'voltmeter' in sim_dict['rec_dev']:
            for voltmeter in net.voltmeters:
                events = nest.GetStatus(voltmeter, "events")[0]
                times = events["times"]
                senders = events["senders"]
                Vms = events["V_m"]

                # append to one ASCII file
                with open(f"{path}/Vm_all.dat", "a") as f:
                    for t, s, v in zip(times, senders, Vms):
                        f.write(f"{s}\t{t:.3f}\t{v:.6f}\n")

                # clear buffer so memory doesn’t explode
                nest.SetStatus(voltmeter, {"n_events": 0})

        # Save ATP
        if 'atpmeter' in sim_dict['rec_dev']:
            for atpmeter in net.atpmeters:
                events = nest.GetStatus(atpmeter, "events")[0]
                times = events["times"]
                senders = events["senders"]
                ATP = events["ATP"]

                # append to one ASCII file
                with open(f"{path}/ATP_all.dat", "a") as f:
                    for t, s, v in zip(times, senders, ATP):
                        f.write(f"{s}\t{t:.3f}\t{v:.6f}\n")

                # clear buffer so memory doesn’t explode
                nest.SetStatus(atpmeter, {"n_events": 0})

        # Save spikes
        for i,spike in enumerate(net.spike_recorders):
            if i!=net.out1 and i!=net.out2:
                events = nest.GetStatus(spike, "events")[0]
                times = events["times"]
                senders = events["senders"]

                # append to one ASCII file
                with open(f"{path}/Spikes_all.dat", "a") as f:
                    for t, s in zip(times, senders):
                        f.write(f"{s}\t{t:.3f}\n")

                # clear buffer so memory doesn’t explode
                nest.SetStatus(spike, {"n_events": 0})
        # Action
        action = net.decode_actions()

        # Step
        obs, firing_rate, done, info = env.step(action)
        firing_rate= np.array(firing_rate)


        if done:
            # print("-----------------------------------------------------------------------------------------------------------------------------CRASH")
            data['Crashes'].append(1)
            obs = env.reset()
            # Pick random subset of electrodes (1 to 8)
            n_sites = np.random.randint(1, len(net.electrodes)+1)
            targets = np.random.choice(net.electrodes, size=n_sites, replace=False)

            # Create a Poisson generator (5 Hz during window)
            pg = nest.Create("poisson_generator", params={
                "rate": 5,  # NEST expects Hz, 5 Hz = 5 spikes/sec
                "start": net.actual_simulation_time,
                "stop": net.actual_simulation_time + 4000
            })

            # Connect generator strongly so it almost always forces spikes
            # The weight acts like the "150 mV" disruptive effect
            for tgt in targets:
                nest.Connect(pg, tgt, syn_spec={"weight": 150.0, "delay": 1.0})
        else:
            data['Crashes'].append(0)
        
        # Save data
        data['frI_Left'].append(sum(fr_left)/(net.n_sensors/2))
        data['frI_Right'].append(sum(fr_right)/(net.n_sensors/2))
        data['frO_Left'].append(round(net.left_fr,2))
        data['frO_Right'].append(round(net.right_fr,2))
        data['v_Left'].append(round(env.velocity_left_wheel,2))
        data['v_Right'].append(round(env.velocity_right_wheel,2))

    
        # Render the environment
        env.render()
        
        # Check memory usage:
        if epoch % 20 == 0:  # every 1 second of sim
            rec = [net.spike_recorders]
            if 'atpmeter' in sim_dict['rec_dev']: rec.append(net.atpmeters)
            if 'voltmeter' in sim_dict['rec_dev']: rec.append(net.voltmeters)
            print_diagnostics(recorders=rec, label=f"after {simulation_time*(epoch+1)} ms")
        """
        Plotter configuration
        """
    
        if round(net.actual_simulation_time-t_init)%1000==0 or epoch==epochs: # Every 10 seconds
            if args.test:
                if epoch < epochs:
                    env.fig.savefig(fr'{path}/Learning_run.png')
                else:
                    env.fig.savefig(fr'{path}/Free_run.png')
            else:
                env.fig.savefig(fr'{path}/Trayectory.png')
            """
            Save data
            """
            print(f'Saving data epoch {epoch}/{total_epochs}')
            # print('Saving FVAC')
            pd.DataFrame(data).to_csv(fr'{path}/dataFVW.csv', index= True, mode='a', header= False)
            for css in data:
                data[css] = []
            epoch_saved = epoch-1
            sim_time_saved = int(net.actual_simulation_time-1)
            
            if net_dict["synapse_model"] is not None and epoch==epochs:
                print('Saving Spikes')
                events = nest.GetStatus(net.spike_recorders[net.out1], "events")[0]
                times = events["times"]
                senders = events["senders"]

                # append to one ASCII file
                with open(f"{path}/Spikes_all.dat", "a") as f:
                    for t, s in zip(times, senders):
                        f.write(f"{s}\t{t:.3f}\n")
                # clear buffer so memory doesn’t explode
                nest.SetStatus(spike, {"n_events": 0})
                events = nest.GetStatus(net.spike_recorders[net.out2], "events")[0]
                times = events["times"]
                senders = events["senders"]

                # append to one ASCII file
                with open(f"{path}/Spikes_all.dat", "a") as f:
                    for t, s in zip(times, senders):
                        f.write(f"{s}\t{t:.3f}\n")
                # clear buffer so memory doesn’t explode
                nest.SetStatus(spike, {"n_events": 0})
                print('Saving weights')
                final_weights = {}
                syn = nest.GetConnections(synapse_model='edstdp_e')
                final_weights['source']= list(nest.GetStatus(syn, "source"))
                final_weights['target']= list(nest.GetStatus(syn, "target"))
                final_weights['final_w']= list(nest.GetStatus(syn, "w"))
                syn = nest.GetConnections(synapse_model='static_i')
                final_weights['source'].extend(list(nest.GetStatus(syn, "source")))
                final_weights['target'].extend(list(nest.GetStatus(syn, "target")))
                final_weights['final_w'].extend(list(nest.GetStatus(syn, "weight")))
                
                pd.DataFrame(final_weights).to_csv(fr"{path}/final_weights.csv",index=False)
                
    env.close()

