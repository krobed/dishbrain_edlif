# -*- coding: utf-8 -*-
#
# network.py
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

"""PyNEST Microcircuit: Network Class
----------------------------------------

Main file of the microcircuit defining the ``Network`` class with functions to
build and simulate the network.

"""

import os
import numpy as np
import pandas as pd
import nest
from utils import helpers
import warnings
import matplotlib.pyplot as plt

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def neurons_near(pos_dict, center, radius):
    return [nid for nid, pos in pos_dict.items() if euclidean(pos, center) <= radius]



class Network:
    """ Provides functions to setup NEST, to create and connect all nodes of
    the network, to simulate, and to evaluate the resulting spike data.

    Instantiating a Network object derives dependent parameters and already
    initializes the NEST kernel.

    Parameters
    ---------
    sim_dict
        Dictionary containing all parameters specific to the simulation
        (see: ``sim_params.py``).
    net_dict
         Dictionary containing all parameters specific to the neuron and
         network models (see: ``network_params.py``).
    stim_dict
        Optional dictionary containing all parameter specific to the stimulus
        (see: ``stimulus_params.py``)

    """

    def __init__(self, sim_dict, net_dict, stim_dict={}):
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict

        # data directory
        self.data_path = sim_dict.get('data_path', None)
        
        if nest.Rank() == 0:
            if os.path.isdir(self.data_path):
                message = '  Directory already existed.'
                if self.sim_dict['overwrite_files']:
                    message += ' Old data will be overwritten.'
            else:
                os.makedirs(self.data_path)
                message = '  Directory has been created.'
            print('Data will be written to: {}\n{}\n'.format(self.data_path,
                                                            message))
            pd.Series(self.sim_dict).to_json(os.path.join(self.data_path,'sim_params.json'))
            pd.Series(self.net_dict).to_json(os.path.join(self.data_path,'net_params.json'))
            pd.Series(self.stim_dict).to_json(os.path.join(self.data_path,'stim_params.json'))

        # derive parameters based on input dictionaries
        self.__derive_parameters()

        # initialize the NEST kernel
        self.__setup_nest()

        self.actual_simulation_time = 0.0
        self.simulation_time = self.sim_dict['t_sim']

    def create(self):
        """ Creates all network nodes.

        Neuronal populations and recording and stimulation devices are created.

        """
        self.__create_neuronal_populations()

        if len(self.sim_dict.get('rec_dev', [])) > 0:
            self.__create_recording_devices()
        if self.net_dict.get('poisson_input', False):
            self.__create_poisson_bg_input()
        if self.stim_dict.get('thalamic_input', False):
            self.__create_thalamic_stim_input()
        if self.stim_dict.get('dc_input', False):
            self.__create_dc_stim_input()
        if self.stim_dict.get('spikes_input', False):
           self.__create_spike_stim_input()

    def connect(self):
        """ Connects the network.

        Recurrent connections among neurons of the neuronal populations are
        established, and recording and stimulation devices are connected.

        The ``self.__connect_*()`` functions use ``nest.Connect()`` calls which
        set up the postsynaptic connectivity.
        Since the introduction of the 5g kernel in NEST 2.16.0 the full
        connection infrastructure including presynaptic connectivity is set up
        afterwards in the preparation phase of the simulation.
        The preparation phase is usually induced by the first
        ``nest.Simulate()`` call.
        For including this phase in measurements of the connection time,
        we induce it here explicitly by calling ``nest.Prepare()``.

        """
        self.__connect_neuronal_populations()

        if len(self.sim_dict.get('rec_dev', [])) > 0:
            self.__connect_recording_devices()
        if self.net_dict.get('poisson_input', False):
            self.__connect_poisson_bg_input()
        if self.stim_dict.get('thalamic_input', False):
            self.__connect_thalamic_stim_input()
        if self.stim_dict.get('dc_input', False):
            self.__connect_dc_stim_input()
        if self.stim_dict.get('spikes_input', False):
           self.__connect_spikes_stim_input()

    def connect_networks(self, net, lateral_dict):
        self.__connect_lateral_neuronal_populations(net, lateral_dict)

    def simulate(self, t_sim):
        """ Simulates the microcircuit.

        Parameters
        ----------
        t_sim
            Simulation time (in ms).

        """
        # if nest.Rank() == 0:
        #     print('Simulating {} ms.'.format(t_sim))

        self.actual_simulation_time += t_sim
        nest.Simulate(t_sim)

    def external_stimulation(self, spike_times):
        """ Delivers external stimulation to the network.

        Parameters
        ----------
        spike_times
            List of spike times (in ms) for each population.
        
        """

        # if nest.Rank() == 0:
        #     print('Delivering external stimulation.')
        # Biphasic, symmetric phases
        # phase_width_ms = 0.5   # each phase
        amplitude_pA = 75
        if len(spike_times[0]) > 0:
            # n_times = np.arange(2 + self.actual_simulation_time, n_pulses * phase_width_ms*2 + net.actual_simulation_time, 2)
            p_stim = nest.Create("spike_generator", params={"spike_times": spike_times[0]})
            # n_stim = nest.Create("spike_generator", params={"spike_times": n_times.tolist()})
            # Connect with a weight that corresponds to desired current amplitude
            nest.Connect(p_stim, self.electrodes[self.in1], syn_spec={"weight": amplitude_pA})  # weight in pA

        if len(spike_times[1]) > 0:
            # n_times = np.arange(2 + self.actual_simulation_time, n_pulses * phase_width_ms*2 + net.actual_simulation_time, 2)
            p_stim = nest.Create("spike_generator", params={"spike_times": spike_times[1]})
            # n_stim = nest.Create("spike_generator", params={"spike_times": n_times.tolist()})
            # Connect with a weight that corresponds to desired current amplitude
            nest.Connect(p_stim, self.electrodes[self.in2], syn_spec={"weight": amplitude_pA})  # weight in pA

                    

    def evaluate(self, raster_plot_interval, firing_rates_interval):
        """ Displays simulation results.

        Creates a spike raster plot.
        Calculates the firing rate of each population and displays them as a
        box plot.

        Parameters
        ----------
        raster_plot_interval
            Times (in ms) to start and stop loading spike times for raster plot
            (included).
        firing_rates_interval
            Times (in ms) to start and stop lading spike times for computing
            firing rates (included).

        Returns
        -------
            None

        """
        if nest.Rank() == 0:
            print('Interval to plot spikes: {} ms'.format(raster_plot_interval))
            if self.sim_dict.get("plot_raster", False):
                helpers.plot_raster(
                    self.data_path,
                    'spike_recorder',
                    raster_plot_interval[0],
                    raster_plot_interval[1],
                    self.net_dict['N_scaling'],
                    self.net_dict['populations'],
                )
            print('Interval to compute firing rates: {} ms'.format(
                firing_rates_interval))
            if self.sim_dict.get("plot_firing_rates", False):
                helpers.firing_rates(
                    self.data_path, 
                    'spike_recorder',
                    firing_rates_interval[0], 
                    firing_rates_interval[1])
                #helpers.boxplot(self.data_path, self.net_dict['populations'])
            if self.sim_dict.get("plot_voltages", False):
                helpers.plot_voltages(
                    self.data_path, 
                    'voltmeter', 
                    firing_rates_interval[0], 
                    firing_rates_interval[1], 
                    self.net_dict['populations'],
                    'spike_recorder' if 'spike_recorder' in self.sim_dict["rec_dev"] else None,
                    self.input_meters.keys()
                )
            if self.sim_dict.get("plot_network", False):
                helpers.plot_network(
                    self.data_path,
                    self.net_dict["populations"], 
                    self.net_dict["conn_weights"],
                    self.stim_dict["conn_weights_th"] if self.stim_dict["thalamic_input"] else None,
                )

    def __derive_parameters(self):
        """
        Derives and adjusts parameters and stores them as class attributes.
        It must be implemented in the derived class.
        """
        raise NotImplementedError

    def __setup_nest(self):
        """ Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        """
        pass

    def __create_neuronal_populations(self):
        """ Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from
        normal distributions dependent on the parameter ``V0_type``.

        The first and last neuron id of each population is written to file.

        It must be implemented in the derived class.
        """
        raise NotImplementedError

    def __create_recording_devices(self):
        """ Creates one recording device of each kind per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.

        """
        
        if nest.Rank() == 0:
            print('Creating recording devices.')
        # Populations spike recorder
        if 'spike_recorder' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('  Creating spike recorders.')
            self.spike_recorders = nest.Create(
                'spike_recorder',
                n= len(self.electrodes))
            
        # Populations voltage recorder
        if 'voltmeter' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('  Creating voltmeters.')
            vm_dict = {
                'interval': self.sim_dict['rec_V_int'],
                'record_to': 'memory',
                'record_from': ['V_m'],
                'label': os.path.join(self.data_path, 'voltmeter')}
            self.voltmeters = nest.Create(
                'voltmeter',
                n=len(self.electrodes),
                params=vm_dict)
        if 'atpmeter' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('  Creating ATP meters.')
            mm_dict = {
                'interval': self.sim_dict['rec_V_int'],
                'record_from': ['ATP'],
                'record_to': 'memory',
                'label': os.path.join(self.data_path, 'atpmeter')}
            self.atpmeters = nest.Create(
                'multimeter',
                n=len(self.electrodes),
                params=mm_dict)

        self.__create_input_recording_devices()

    def __create_input_recording_devices(self):
        # Input meters
        self.input_meters = {}
        # Poisson input meter
        if self.net_dict.get('poisson_input', None):
            if nest.Rank() == 0:
                print('  Creating poisson input meters.')
            pg_dict = {
                'record_to': 'ascii',
                'label': os.path.join(self.data_path, 'poisson_sr')}
            poisson_sr_meters = nest.Create('spike_recorder',
                                            n=self.num_pops,
                                            params=pg_dict)
            self.input_meters["poisson_sr"] = poisson_sr_meters
        # Thalamic input meter
        if self.stim_dict.get('thalamic_input', None):
            if nest.Rank() == 0:
                print('  Creating thalamic multimeters.')
            th_dict = {
                'record_to': 'ascii',
                'label': os.path.join(self.data_path, 'thalamic_sr')}
            thalamic_sr_meter = nest.Create('spike_recorder',
                                            n=self.num_pops,
                                            #n=self.stim_dict["num_th_neurons"],
                                            params=th_dict)
            self.input_meters["thalamic_sr"] = thalamic_sr_meter 
        # DC input meter
        if self.stim_dict.get('dc_input', None):
            if nest.Rank() == 0:
                print('  Creating DC multimeters.')
            dc_dict = {
                'interval': self.sim_dict['rec_V_int'],
                'record_to': 'ascii',
                'record_from': ['I'],
                'label': os.path.join(self.data_path, 'dc_input')}
            dc_meters = nest.Create('multimeter',
                            n=self.num_pops,
                            params=dc_dict
            )
            self.input_meters["dc_input"] = dc_meters

    def __create_poisson_bg_input(self):
        """ Creates the Poisson generators for ongoing background input if
        specified in ``network_params.py``.

        If ``poisson_input`` is ``False``, DC input is applied for compensation
        in ``create_neuronal_populations()``.

        """
        if nest.Rank() == 0:
            print('Creating Poisson generators for background input.')

        self.poisson_bg_input = nest.Create(
            'poisson_generator',
            n=self.num_pops)
        self.poisson_bg_input.rate = \
            self.net_dict['bg_rate'] * self.ext_indegrees

    def __create_thalamic_stim_input(self):
        """ Creates the thalamic neuronal population if specified in
        ``stim_dict``.

        Each neuron of the thalamic population is supposed to transmit the same
        Poisson spike train to all of its targets in the cortical neuronal population,
        and spike trains elicited by different thalamic neurons should be statistically
        independent.
        In NEST, this is achieved with a single Poisson generator connected to all
        thalamic neurons which are of type ``parrot_neuron``;
        Poisson generators send independent spike trains to each of their targets and
        parrot neurons just repeat incoming spikes.        
        
        Note that the number of thalamic neurons is not scaled with
        ``N_scaling``.

        """
        if nest.Rank() == 0:
            print('Creating thalamic input for external stimulation.')

        self.thalamic_population = nest.Create(
            'parrot_neuron', n=self.stim_dict['num_th_neurons'])

        self.poisson_th = nest.Create('inhomogeneous_poisson_generator')
        self.poisson_th.set(
            # rate=self.stim_dict['th_rate'],
            # start=self.stim_dict['th_start'],
            # stop=(self.stim_dict['th_start'] + self.stim_dict['th_duration']))
            rate_values=[self.stim_dict['th_rate'], 2*self.stim_dict['th_rate']],
            rate_times=[self.stim_dict['th_start'], 2*self.stim_dict['th_start']])

    def __create_dc_stim_input(self):
        """ Creates DC generators for external stimulation if specified
        in ``stim_dict``.

        The final amplitude is the ``stim_dict['dc_amp'] * net_dict['K_ext']``.

        """
        dc_amp_stim = self.stim_dict['dc_amp'] #* self.net_dict['K_ext']

        if nest.Rank() == 0:
            print('Creating DC generators for external stimulation.')

        dc_dict = {
            'amplitude': dc_amp_stim,
            'start': self.stim_dict['dc_start'],
            'stop': self.stim_dict['dc_start'] + self.stim_dict['dc_dur']}
        self.dc_stim_input = nest.Create(
            'dc_generator', 
            n=self.num_pops, 
            params=dc_dict)

    def __create_spike_stim_input(self, sg_dict={}):
        """ Creates spike generators for external stimulation if specified
        in ``stim_dict``.

        Parameters
        ----------
        spike_times
            List of spike times (in ms) for each population.

        """
        if nest.Rank() == 0:
            print('Creating spike generators for external stimulation.')
        self.spikes_stim_input = nest.Create(
            'spike_generator', 
            n=self.num_pops,
            params=sg_dict)

    def __connect_neuronal_populations(self):
        """ Creates the recurrent connections between neuronal populations. 

            It must be implemented in the derived class.
        """
        raise NotImplementedError

    def __connect_lateral_neuronal_populations(self, net, lateral_dict):
        """ Creates the recurrent connections between neuronal populations. 

            It must be implemented in the derived class.
        """
        raise NotImplementedError

    def __connect_recording_devices(self):
        """ Connects the recording devices to the microcircuit."""
        if nest.Rank == 0:
            print('Connecting recording devices.')

        if 'spike_recorder' in self.sim_dict['rec_dev']:
            for i,electrode in enumerate(self.electrodes):
                nest.Connect(electrode,self.spike_recorders[i])
            
        if 'voltmeter' in self.sim_dict['rec_dev']:
            for i,electrode in enumerate(self.electrodes):
                nest.Connect(self.voltmeters[i], electrode)

        if 'atpmeter' in self.sim_dict['rec_dev']:
            for i,electrode in enumerate(self.electrodes):
                nest.Connect(self.atpmeters[i], electrode)
            
    def __connect_poisson_bg_input(self):
        """ Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print('Connecting Poisson generators for background input.')

        for i, target_pop in enumerate(self.pops):
            conn_dict_poisson = {'rule': 'all_to_all'}

            syn_dict_poisson = {
                'synapse_model': 'static_synapse',
                'weight': self.weight_ext,
                'delay': self.net_dict['delay_poisson']}

            nest.Connect(
                self.poisson_bg_input[i], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson)

    def __connect_thalamic_stim_input(self):
        """ Connects the thalamic input to the neuronal populations.

            It must be implemented in the derived class.
        """
        raise NotImplementedError

    def __connect_dc_stim_input(self):
        """ Connects the DC generators to the neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting DC generators.')

        for i, target_pop in enumerate(self.pops):
            nest.Connect(self.dc_stim_input[i], target_pop)

    def __connect_spikes_stim_input(self):
        """ Connects the spikes input to the neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting spikes input.')

        for i, target_pop in enumerate(self.pops):
            nest.Connect(self.spikes_stim_input[i], target_pop)
