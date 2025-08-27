import os
import numpy as np
import nest
from utils import helpers
import warnings

from . import network

class Network(network.Network):
    """ Provides functions to setup NEST, to create and connect all nodes of
    the network, to simulate, and to evaluate the resulting spike data.

    Instantiating a Network object derives dependent parameters and already
    initializes the NEST kernel.

    Implementation:
    ------------
    RODNEY J. DOUGLAS*t AND KEVAN A. C. MARTIN* 
    A FUNCTIONAL MICROCIRCUIT FOR CAT VISUAL CORTEX

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
    def __init__(self, sim_dict, net_dict, stim_dict=None):
        super().__init__(sim_dict, net_dict, stim_dict)

    def create(self):
        """ Creates all network nodes.

        Neuronal populations and recording and stimulation devices are created.

        """
        super().create()

    def connect(self):
        """ Connects all network nodes.

        Connects all neuronal populations and recording and stimulation
        devices.

        """
        super().connect()

    def simulate(self, t_sim):
        """ Simulates the network.

        Simulates the network for the specified duration.

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
        self.population_names = self.net_dict["populations"]
        self.num_pops = len(self.population_names)
        self.num_neurons = self.net_dict["full_num_neurons"]
        self.conn_weights = self.net_dict["conn_weights"]
        self.conn_weights_th = self.stim_dict["conn_weights_th"]
        self.conn_delays = self.net_dict["conn_delays"]
        self.conn_delays_th = self.stim_dict["conn_delays_th"]

    def __setup_nest(self):
        """ Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        """
        super().__setup_nest()

    def __create_neuronal_populations(self):
        """ Creates the neuronal populations.

        Creates the neuronal populations and stores them as class attributes.

        """
        if nest.Rank() == 0:
            print('Creating neuronal populations.')

        self.pops = []
        for i in range(self.num_pops):
            # Get associated neuron params from the specific population
            neuron_params = self.net_dict["neuron_params"][self.population_names[i]]
            # Create the population
            population = nest.Create(self.net_dict["neuron_model"],
                                    self.num_neurons[i])
            population.set(**neuron_params)
            # Store the population
            self.pops.append(population)

        # write node ids to file
        if nest.Rank() == 0:
            fn = os.path.join(self.data_path, 'population_nodeids.dat')
            with open(fn, 'w+') as f:
                for pop in self.pops:
                    f.write('{} {}\n'.format(pop[0].global_id,
                                             pop[-1].global_id))

    def __create_recording_devices(self):
        """ Creates one recording device of each kind per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.

        """
        super().__create_recording_devices()

    def __create_input_recording_devices(self):
        """ Creates one recording device of each kind per input population.

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
        """ Creates the connections between neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting neuronal populations recurrently.')

        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                #conn_dict_rec = {
                #}
                syn_dict = {
                    'synapse_model': 'static_synapse',
                    'weight': self.conn_weights[i][j],
                    'delay': self.conn_delays[i][j]
                }

                nest.Connect(
                    source_pop, target_pop,
                    #conn_spec=conn_dict_rec,
                    syn_spec=syn_dict)

    def __connect_recording_devices(self):
        """ Connects the recording devices to the microcircuit."""
        super().__connect_recording_devices()

    def __connect_poisson_bg_input(self):
        """ Connects the Poisson generators to the microcircuit."""
        super().__connect_poisson_bg_input()

    def __connect_thalamic_stim_input(self):
        """ Connects the thalamic input to the neuronal populations."""
        # connect Poisson input to thalamic population
        if nest.Rank() == 0:
            print('Connecting thalamic input.')

        nest.Connect(self.poisson_th, self.thalamic_population)

        # connect thalamic population to neuronal populations
        for i, target_pop in enumerate(self.pops):
            #conn_dict_th = {   
            #}
            syn_dict_th = {
                'weight': self.conn_weights_th[i],
                'delay': self.conn_delays_th[i]
            }
            nest.Connect(self.thalamic_population, target_pop,
                            #conn_spec=conn_dict_th,
                            syn_spec=syn_dict_th)

    def __connect_dc_stim_input(self):
        """ Connects the DC generators to the neuronal populations."""
        super().__connect_dc_stim_input()