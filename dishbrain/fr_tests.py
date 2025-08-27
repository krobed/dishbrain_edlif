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

###############################################################################
# Import the necessary modules and start the time measurements.
import nest
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from random import randint
import os

from environment import CustomCircleCarEnv
from utils.nest_utils import setup_nest

t=round(time.time())
path= fr"{os.path.dirname(os.path.abspath(__file__))}/results/sim_results/{t}"
if not os.path.exists(path):
    os.makedirs(path)

def select_sensor(lista, n_sensors):
    x= input(f'Ingrese número de sensor entre 1 y {n_sensors}: ')
    try:
        int(x)
    except:
        print('Ingrese un número válido')
        select_sensor(lista,n_sensors)
    else:
        if int(x)<=n_sensors+1 and int(x)>=1:
            n= int(x)
            lista[n] =3
            return lista, int(x)
        else:
            select_sensor(lista,n_sensors)


args = argparse.ArgumentParser()
args.add_argument('--microcircuit', type=str, default="edsnn")

if __name__ == '__main__':
    args = args.parse_args()

    if args.microcircuit == "douglas":
        from assets.douglas.stimulus_params import stim_dict # type: ignore
        from assets.douglas.network_params import net_dict # type: ignore
        from assets.douglas.sim_params import sim_dict   # type: ignore
        from network import network_douglas as network
    elif args.microcircuit == "potjans_diesmann":
        from assets.potjans_diesmann.stimulus_params import stim_dict # type: ignore
        from assets.potjans_diesmann.network_params import net_dict # type: ignore
        from assets.potjans_diesmann.sim_params import sim_dict # type: ignore
        from network import network_potjans_diesmann as network
    elif args.microcircuit == "edsnn":
        from assets.edsnn.stimulus_params import stim_dict
        from assets.edsnn.network_params import net_dict
        from assets.edsnn.sim_params import sim_dict
        from network import network_edsnn as network

    time_start = time.time()

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
        wr = nest.Create("weight_recorder")
        nest.CopyModel(
            net_dict["synapse_model_lib"], 
            net_dict["synapse_model"],
            {
                "weight_recorder": wr,
            }
        )

    # Environment
    env = CustomCircleCarEnv(
        # width=20.0,
        # height=20.0,
    )

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
    obs = env.reset()

    # Simulation parameters
    epochs = 1000
    simulation_time = sim_dict['t_sim']
    firing_rate = [0]*len(net.pops[0])
    fig, ax = plt.subplots(5, 1, figsize= (8,10))
    print(nest.GetDefaults(net_dict["neuron_model"])['recordables'])
    input_firing_rates, left_firing_rates, right_firing_rates = [], [], []
    left_vel, right_vel = [], []
    input_weights = []
    left_weights = []
    right_weights = []
    # firing_rate, sensor = select_sensor(np.zeros(len(net.pops[0])),len(net.pops[0]))
    print(firing_rate)
    for epoch in range(epochs):
        print(f"Epoch {epoch}, firing_rate: {sum(firing_rate)}")
        print("actual_time:", net.actual_simulation_time)
        # Generate spike times
        #firing_rate = np.random.randint(1, 20)
        #spike_times = np.linspace(0, simulation_time, int(simulation_time * firing_rate), endpoint=False) 
        spike_times = [] #+ net.actual_simulation_time
            
        #spike_times = np.random.uniform(0, simulation_time, int(simulation_time * firing_rate))
        # Generate spike times       
        #firing_rate = 15
        input_firing_rates.append(firing_rate)
        for i in firing_rate:
            # print(net.actual_simulation_time)
            # print(simulation_time)
            spike_times.append(np.linspace(
                            net.actual_simulation_time, 
                            simulation_time + net.actual_simulation_time, 
                            int((simulation_time) * i)))
        #print(spike_times)
        #spike_times = []
        #print("input spike_times:", spike_times)
        
        # External stimulation
        net.external_stimulation(
            spike_times=spike_times)

        if net_dict["synapse_model"] is not None:
            syn_l = nest.GetConnections(
                source=net.pops[1], 
                synapse_model=net_dict["synapse_model"])
            left_weight = [i["w"] for i in nest.GetStatus(syn_l)]
            syn_r = nest.GetConnections(
                source=net.pops[2], 
                synapse_model=net_dict["synapse_model"])
            right_weight = [i["w"] for i in nest.GetStatus(syn_r)]
            syn = nest.GetConnections(
                source=net.pops[0], 
                synapse_model=net_dict["synapse_model"])
            initial_weight = [i["w"] for i in nest.GetStatus(syn)]

        # Simulate
        net.simulate(
            t_sim=simulation_time)
        
        # if epoch%30==0:
        #     firing_rate[sensor]= 3
        # elif epoch%40==0:
        #     firing_rate*=0

        if net_dict["synapse_model"] is not None:
            updated_weight = [i["w"] for i in nest.GetStatus(syn)]
            # print("initial_weight", initial_weight)
            # print("updated_weight", updated_weight)
        # Action
        action = net.decode_actions()
        # Step
        obs, _, done, info = env.step(action, len(net.pops[0]))

        # Plots
        input_weights.append(initial_weight)
        left_weights.append(left_weight)
        right_weights.append(right_weight)
        left_firing_rates.append(net.left_fr)
        right_firing_rates.append(net.right_fr)
        left_vel.append(env.velocity_left_wheel)
        right_vel.append(env.velocity_right_wheel)
        if epoch%5 == 0 or done:
            s_times = nest.GetStatus(net.spike_recorders[0], 'events')[0]["times"]
            voltages = nest.GetStatus(net.voltmeters[0], 'events')[0]['V_m']
            ATP = nest.GetStatus(net.atpmeters[0], 'events')[0]['ATP']
            ax[0].plot(voltages, "C0")
            ax[0].plot(s_times, -67 * np.ones(len(s_times)), "ok")
            ax[0].set_title("voltages")
            ax[1].plot(ATP, "C0")
            ax[1].set_title("ATP")
            ax[2].plot(input_firing_rates, "k")
            ax[2].plot(left_firing_rates, "C0", label="left")
            ax[2].plot(right_firing_rates, "C1", label="right")
            ax[2].set_title("firing rates")
            ax[3].plot(left_vel, "C0", label="left")
            ax[3].plot(right_vel, "C1", label="right")
            ax[3].set_title("velocities")
            ax[4].plot(input_weights, "C0")
            # ax[4].plot(right_weights, "C0")
            # ax[4].plot(left_weights, "C1")
            ax[4].set_title("weights")
            if epoch == 0:
                ax[0].grid()
                ax[1].grid()
                ax[2].grid()
                ax[3].grid()
                ax[4].grid()
                ax[2].legend()
                ax[3].legend()
            fig.tight_layout()
            start = epoch

        # Render the environment
        env.render()
        # Reset if done
        if done:
            #print("-----------------------------------------------------------------------------------------------------------------------------RESET")
            obs = env.reset()
            env.fig.savefig(f'results/sim_results/{t}/Robot epoch {epoch}.png')
            fig.savefig(f'results/sim_results/{t}/Data epoch {epoch}.png')
        #time.sleep(0.1)
        time_simulate = time.time()
    env.fig.savefig(f'results/sim_results/{t}/Robot epoch {epoch}.png')
    fig.savefig(f'results/sim_results/{t}/Data epoch {epoch}.png')
    env.close()


    # wr= wr.get("events")
    # mult = net.multimeter
    # mult = mult.get('events')
    # spike = net.spike_recorders
    # spike = spike.get("events")
    # plt.figure()
    # plt.xlabel("time (ms)")
    # plt.ylabel("Weights")
    # plt.plot(wr['times'], wr['weights'], '.')


    # plt.figure()
    # plt.xlabel("time (ms)")
    # plt.ylabel("ATP")
    # c = ['r','k','b','g']
    # for i in range(len(mult)):
    #     plt.plot(mult[i]['times'], mult[i]['ATP'], c[i])

    # plt.figure()
    # plt.xlabel("time (ms)")
    # plt.ylabel("Spikes")
    # c = ['r','k','b','g']
    # for i in range(len(spike)):
    #     plt.plot(spike[i]['times'],[1 for j in range(len(spike[i]['times']))], '.')
    # plt.show()
    ###############################################################################
    # Plot a spike raster of the simulated neurons and a box plot of the firing
    # rates for each population.
    # For visual purposes only, spikes 100 ms before and 100 ms after the thalamic
    # stimulus time are plotted here by default.
    # The computation of spike rates discards the presimulation time to exclude
    # initialization artifacts.

    #raster_plot_interval = np.array([stim_dict['th_start'] - 100.0,
    #                                stim_dict['th_start'] + 100.0 + sim_dict["t_sim"]])
    # raster_plot_interval = np.array([sim_dict['t_presim'], sim_dict["t_sim"]])
    # firing_rates_interval = np.array([sim_dict['t_presim'], sim_dict["t_sim"]])
    # net.evaluate(raster_plot_interval, firing_rates_interval)
    # time_evaluate = time.time()


    ###############################################################################
    # Histogramas de spikes

    #import src.histogram_single_microcircuit as hist_spikes
    #data_path = sim_dict.get('data_path', None)
    #archivos_spike_recorder = hist_spikes.select_spike_recorder_files(data_path)
    #hist_spikes.apliccation_metrics(data_path, archivos_spike_recorder)
 

    ###############################################################################
    # Summarize time measurements. Rank 0 usually takes longest because of the
    # data evaluation and print calls.

    # print(
    #     '\nTimes of Rank {}:\n'.format(
    #         nest.Rank()) +
    #     '  Total time:          {:.3f} s\n'.format(
    #         time_evaluate -
    #         time_start) +
    #     '  Time to initialize:  {:.3f} s\n'.format(
    #         time_network -
    #         time_start) +
    #     '  Time to create:      {:.3f} s\n'.format(
    #         time_create -
    #         time_network) +
    #     '  Time to connect:     {:.3f} s\n'.format(
    #         time_connect -
    #         time_create) +
    #     '  Time to presimulate: {:.3f} s\n'.format(
    #         time_presimulate -
    #         time_connect) +
    #     '  Time to simulate:    {:.3f} s\n'.format(
    #         time_simulate -
    #         time_presimulate) +
    #     '  Time to evaluate:    {:.3f} s\n'.format(
    #         time_evaluate -
    #         time_simulate))

    # plt.show()