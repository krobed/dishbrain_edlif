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

from . import network

class Network(network.Network):
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
        super().__init__(sim_dict, net_dict, stim_dict)

    def create(self):
        """ Creates all network nodes.

        Neuronal populations and recording and stimulation devices are created.

        """
        super().create()

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
        super().connect()

    def connect_networks(self, net, lateral_dict):
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
        super().connect_networks(net, lateral_dict)


    def simulate(self, t_sim):
        """ Simulates the microcircuit.

        Parameters
        ----------
        t_sim
            Simulation time (in ms).

        """
        super().simulate(t_sim)

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
        super().evaluate(raster_plot_interval, firing_rates_interval)


    def __derive_parameters(self):
        """
        Derives and adjusts parameters and stores them as class attributes.
        """
        self.num_pops = len(self.net_dict['populations'])

        # total number of synapses between neuronal populations before scaling
        full_num_synapses = helpers.num_synapses_from_conn_probs(
            self.net_dict['conn_probs'],
            self.net_dict['full_num_neurons'],
            self.net_dict['full_num_neurons'])

        # scaled numbers of neurons and synapses
        self.num_neurons = np.round((self.net_dict['full_num_neurons'] *
                                     self.net_dict['N_scaling'])).astype(int)
        self.num_synapses = np.round((full_num_synapses *
                                      self.net_dict['N_scaling'] *
                                      self.net_dict['K_scaling'])).astype(int)
        self.ext_indegrees = np.round((self.net_dict['K_ext'] *
                                       self.net_dict['K_scaling'])).astype(int)

        # conversion from PSPs to PSCs
        PSC_over_PSP = helpers.postsynaptic_potential_to_current(
            self.net_dict['neuron_params']['C_m'],
            self.net_dict['neuron_params']['tau_m'],
            self.net_dict['neuron_params']['tau_syn'])
        PSC_matrix_mean = self.net_dict['PSP_matrix_mean'] * PSC_over_PSP
        PSC_ext = self.net_dict['PSP_exc_mean'] * PSC_over_PSP

        # DC input compensates for potentially missing Poisson input
        if self.net_dict['poisson_input']:
            DC_amp = np.zeros(self.num_pops)
        else:
            if not self.net_dict['dc_compensation']:
                DC_amp = np.zeros(self.num_pops)
            else:
                if nest.Rank() == 0:
                    warnings.warn('DC input created to compensate missing Poisson input.\n')
                DC_amp = helpers.dc_input_compensating_poisson(
                    self.net_dict['bg_rate'], self.net_dict['K_ext'],
                    self.net_dict['neuron_params']['tau_syn'],
                    PSC_ext)

        # adjust weights and DC amplitude if the indegree is scaled
        if self.net_dict['K_scaling'] != 1:
            PSC_matrix_mean, PSC_ext, DC_amp = \
                helpers.adjust_weights_and_input_to_synapse_scaling(
                    self.net_dict['full_num_neurons'],
                    full_num_synapses, 
                    self.net_dict['K_scaling'],
                    PSC_matrix_mean, 
                    PSC_ext,
                    self.net_dict['neuron_params']['tau_syn'],
                    self.net_dict['full_mean_rates'],
                    DC_amp,
                    self.net_dict['poisson_input'],
                    self.net_dict['bg_rate'], self.net_dict['K_ext']
                )

        # store final parameters as class attributes
        self.weight_matrix_mean = PSC_matrix_mean
        self.weight_ext = PSC_ext
        self.DC_amp = DC_amp

        # thalamic input
        if self.stim_dict.get('thalamic_input', False):
            num_th_synapses = helpers.num_synapses_from_conn_probs(
                self.stim_dict['conn_probs_th'],
                self.stim_dict['num_th_neurons'],
                self.net_dict['full_num_neurons'])[0]
            self.weight_th = self.stim_dict['PSP_th'] * PSC_over_PSP
            if self.net_dict['K_scaling'] != 1:
                num_th_synapses *= self.net_dict['K_scaling']
                self.weight_th /= np.sqrt(self.net_dict['K_scaling'])
            self.num_th_synapses = np.round(num_th_synapses).astype(int)

        if nest.Rank() == 0:
            message = ''
            if self.net_dict['N_scaling'] != 1:
                message += \
                    'Neuron numbers are scaled by a factor of {:.3f}.\n'.format(
                        self.net_dict['N_scaling'])
            if self.net_dict['K_scaling'] != 1:
                message += \
                    'Indegrees are scaled by a factor of {:.3f}.'.format(
                        self.net_dict['K_scaling'])
                message += '\n  Weights and DC input are adjusted to compensate.\n'
            print(message)

    def __setup_nest(self):
        """ Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        """
        super().__setup_nest()

    def __create_neuronal_populations(self):
        """ Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from
        normal distributions dependent on the parameter ``V0_type``.

        The first and last neuron id of each population is written to file.
        """
        if nest.Rank() == 0:
            print('Creating neuronal populations.')

        self.pops = []
        for i in np.arange(self.num_pops):
            population = nest.Create(self.net_dict['neuron_model'],
                                     self.num_neurons[i])

            population.set(
                tau_syn_ex=self.net_dict['neuron_params']['tau_syn'],
                tau_syn_in=self.net_dict['neuron_params']['tau_syn'],
                E_L=self.net_dict['neuron_params']['E_L'],
                V_th=self.net_dict['neuron_params']['V_th'],
                V_reset=self.net_dict['neuron_params']['V_reset'],
                t_ref=self.net_dict['neuron_params']['t_ref'],
                I_e=self.DC_amp[i])

            if self.net_dict['V0_type'] == 'optimized':
                population.set(V_m=nest.random.normal(
                    self.net_dict['neuron_params']['V0_mean']['optimized'][i],
                    self.net_dict['neuron_params']['V0_std']['optimized'][i]))
            elif self.net_dict['V0_type'] == 'original':
                population.set(V_m=nest.random.normal(
                    self.net_dict['neuron_params']['V0_mean']['original'],
                    self.net_dict['neuron_params']['V0_std']['original']))
            else:
                raise ValueError(
                    'V0_type is incorrect. ' +
                    'Valid options are "optimized" and "original".')

            self.pops.append(population)

        # write node ids to file
        if nest.Rank() == 0:
            fn = os.path.join(self.data_path, 'population_nodeids.dat')
            with open(fn, 'a') as f:
                for pop in self.pops:
                    f.write('{} {}\n'.format(pop[0].global_id,
                                             pop[-1].global_id))

    def __create_recording_devices(self):
        """ Creates one recording device of each kind per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.

        """
        super().__create_recording_devices()

    def __create_input_recording_devices(self):
        """ Creates one input recording device per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.

        """
        super().__create_input_recording_devices()

    def __create_poisson_bg_input(self):
        """ Creates the Poisson generators for ongoing background input if
        specified in ``network_params.py``.

        If ``poisson_input`` is ``False``, DC input is applied for compensation
        in ``create_neuronal_populations()``.

        """
        super().__create_poisson_bg_input()

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
        super().__create_thalamic_stim_input()

    def __create_dc_stim_input(self):
        """ Creates DC generators for external stimulation if specified
        in ``stim_dict``.

        The final amplitude is the ``stim_dict['dc_amp'] * net_dict['K_ext']``.

        """
        super().__create_dc_stim_input()

    def __connect_neuronal_populations(self):
        """ Creates the recurrent connections between neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting neuronal populations recurrently.')

        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                if self.num_synapses[i][j] >= 0.:
                    conn_dict_rec = {
                        'rule': 'fixed_total_number',
                        'N': self.num_synapses[i][j]}

                    if self.weight_matrix_mean[i][j] < 0:
                        w_min = np.NINF
                        w_max = 0.0
                    else:
                        w_min = 0.0
                        w_max = np.Inf

                    syn_dict = {
                        'synapse_model': 'static_synapse',
                        'weight': nest.math.redraw(
                            nest.random.normal(
                                mean=self.weight_matrix_mean[i][j],
                                std=abs(self.weight_matrix_mean[i][j] *
                                        self.net_dict['weight_rel_std'])),
                            min=w_min,
                            max=w_max),
                        'delay': nest.math.redraw(
                            nest.random.normal(
                                mean=self.net_dict['delay_matrix_mean'][i][j],
                                std=(self.net_dict['delay_matrix_mean'][i][j] *
                                     self.net_dict['delay_rel_std'])),
                            min=nest.resolution,
                            max=np.Inf)}

                    nest.Connect(
                        source_pop, target_pop,
                        conn_spec=conn_dict_rec,
                        syn_spec=syn_dict)

    def __connect_lateral_neuronal_populations(self, net, lateral_dict):
        """ TODO: Creates the recurrent connections between neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting neuronal populations recurrently.')

        full_num_synapses = helpers.num_synapses_from_conn_probs(
            lateral_dict["conn_probs"],
            self.net_dict['full_num_neurons'],
            net.net_dict['full_num_neurons'])
        
        num_synapses = np.round((full_num_synapses *
                                  lateral_dict['N_scaling'] *
                                  lateral_dict['K_scaling'])).astype(int)
        
        # num_synapses = np.array([[454998, 223236, 202536,  96709,  32936,      0,  22714,      0],
        #                         [174437,  50188,  41053,  16901,  22212,      0,   3535,      0],
        #                         [ 35037,   7566, 244828, 174136,   7145,     70, 146244,      0],
        #                         [ 81143,    928,  99335,  52233,    878,      0,  88109,      0],
        #                         [106136,  18171,  55078,   1519,  20407,  24079,  14390,      0],
        #                         [ 12414,   1694,   6077,    129,   3196,   4304,   1324,      0],
        #                         [ 46812,   5561,  67276,  13202,  41122,   3050,  83726, 108277],
        #                         [ 22608,    172,   2200,     81,   4016,    252,  28884,  13543]])
        
        # conversion from PSPs to PSCs
        PSC_over_PSP = helpers.postsynaptic_potential_to_current(
            net.net_dict['neuron_params']['C_m'],
            net.net_dict['neuron_params']['tau_m'],
            net.net_dict['neuron_params']['tau_syn']
        )
        PSC_matrix_mean = net.net_dict['PSP_matrix_mean'] * PSC_over_PSP
    
        # adjust weights and DC amplitude if the indegree is scaled
        if net.net_dict['K_scaling'] != 1:
            PSC_matrix_mean /= np.sqrt(net.net_dict['K_scaling'])

        weight_matrix_mean = PSC_matrix_mean
        #weight_rel_std = 0.1
        
        # delay_matrix_mean = np.array([[1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75],
        #                             [1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75],
        #                             [1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75],
        #                             [1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75],
        #                             [1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75],
        #                             [1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75],
        #                             [1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75],
        #                             [1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75, 1.5 , 0.75]])
        
        #delay_rel_std = 0.5
        
        for i, target_pop in enumerate(net.pops):
            for j, source_pop in enumerate(self.pops):
                if num_synapses[i][j] >= 0.:
                    conn_dict_rec = {
                        'rule': 'fixed_total_number',
                        'N': num_synapses[i][j]}  

                    if weight_matrix_mean[i][j] < 0:
                        w_min = np.NINF
                        w_max = 0.0
                    else:
                        w_min = 0.0
                        w_max = np.Inf
                
                    syn_dict = {
                        'synapse_model': 'static_synapse',
                        'weight': nest.math.redraw(
                            nest.random.normal(
                                mean=weight_matrix_mean[i][j],
                                std=abs(weight_matrix_mean[i][j] * net.net_dict["weight_rel_std"])),
                                min=w_min,
                                max=w_max),
                        'delay': nest.math.redraw(
                            nest.random.normal(
                                mean=net.net_dict["delay_matrix_mean"][i][j],
                                std=(net.net_dict["delay_matrix_mean"][i][j] * net.net_dict["delay_rel_std"])),
                                min=nest.resolution,
                                max=np.Inf)}

                    #print(source_pop, target_pop)
                    nest.Connect(
                        source_pop, target_pop,
                        conn_spec=conn_dict_rec,
                        syn_spec=syn_dict
                    )


    def __connect_recording_devices(self):
        """ Connects the recording devices to the microcircuit."""
        super().__connect_recording_devices()

    def __connect_poisson_bg_input(self):
        """ Connects the Poisson generators to the microcircuit."""
        super().__connect_poisson_bg_input()

    def __connect_thalamic_stim_input(self):
        """ Connects the thalamic input to the neuronal populations."""
        if nest.Rank() == 0:
            print('Connecting thalamic input.')

        # connect Poisson input to thalamic population
        nest.Connect(self.poisson_th, self.thalamic_population)

        # connect thalamic population to neuronal populations
        for i, target_pop in enumerate(self.pops):
            conn_dict_th = {
                'rule': 'fixed_total_number',
                'N': self.num_th_synapses[i]}

            syn_dict_th = {
                'weight': nest.math.redraw(
                    nest.random.normal(
                        mean=self.weight_th,
                        std=self.weight_th * self.net_dict['weight_rel_std']),
                    min=0.0,
                    max=np.Inf),
                'delay': nest.math.redraw(
                    nest.random.normal(
                        mean=self.stim_dict['delay_th_mean'],
                        std=(self.stim_dict['delay_th_mean'] *
                             self.stim_dict['delay_th_rel_std'])),
                    min=nest.resolution,
                    max=np.Inf)}

            nest.Connect(
                self.thalamic_population, target_pop,
                conn_spec=conn_dict_th, syn_spec=syn_dict_th)

    def __connect_dc_stim_input(self):
        """ Connects the DC generators to the neuronal populations. """
        super().__connect_dc_stim_input()
