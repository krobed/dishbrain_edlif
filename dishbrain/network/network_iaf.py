import os
import numpy as np
import nest
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
        self.left_spike_times = []
        self.right_spike_times = []
        self.n_sensors=64
        self.max_fr= self.num_neurons[0]*0.4

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

    def simulate(self,
            t_sim):
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

    def external_stimulation(self, 
            spike_times):
        """ Stimulates the network externally.

        Stimulates the network externally for the specified duration.

        """
        super().external_stimulation(
            spike_times=spike_times
        )

    def decode_actions(self):
        """ Decodes the firing rates of the left and right wheel populations
        """
        def recorder_to_times(sr):
            """
            Transforms the spike recorder to spike times 
            """
            status = nest.GetStatus(sr)
            # print(status)
            spike_times = []
            if status[0]["n_events"] != 0:
                spike_times = status[0]["events"]["times"]
            return spike_times
        
        
        left_sr = self.spike_recorders[-self.n_recorders//2:]
        right_sr = self.spike_recorders[-self.n_recorders:-self.n_recorders//2]
        
        left_spike_times = recorder_to_times(left_sr)
        
        right_spike_times = recorder_to_times(right_sr)
        # print(left_spike_times)
        # print(right_spike_times)
        # Decode actions from spike times
        left_velocity, self.left_fr = self.__decode_action(left_spike_times)
        right_velocity, self.right_fr = self.__decode_action(right_spike_times)
        if self.right_fr+self.left_fr>self.max_fr:
                self.max_fr=self.left_fr+self.right_fr
        if self.left_fr>self.right_fr:
            left_velocity = -1
        if self.left_fr<self.right_fr:
            right_velocity = -1
        return [left_velocity, right_velocity]
    
    def __decode_action(self, spike_times):
        """ Decodes the firing rates of the wheel populations to a velocity.
        """
        # Convert spike times to Quantity objects for Elephant
        if len(spike_times) != 0:
            # print(spike_times)
            spike_times1 = spike_times[
                spike_times >= (self.actual_simulation_time - self.simulation_time)]
            # print(spike_times)
            firing_rate= (len(spike_times1))*1000/self.simulation_time
        else:
            firing_rate = 0.0
        # time, fr = firing_rate(spike_times, self.actual_simulation_time, step= 0.1, time_window= 20.)
        # Exponential function to decode firing rate to velocity
        def firing_rate_to_velocity(firing_rate, k=1):
            print(firing_rate)
            return  np.exp(-(firing_rate/50)**2) + 1

        # Decode firing rates to velocities
        wheel_velocity = firing_rate_to_velocity(firing_rate, k=1)
        
        # Ensure velocities are within [0, 1]
        #wheel_velocity = np.clip(wheel_velocity, 0, 1)

        return wheel_velocity, firing_rate

    def __derive_parameters(self):
        """
        Derives and adjusts parameters and stores them as class attributes.
        """
        self.population_names = self.net_dict["populations"]
        self.num_pops = len(self.population_names)
        self.num_neurons = self.net_dict["full_num_neurons"]
        self.synapse_model = self.net_dict["synapse_model"]
        self.conn_weights = self.net_dict["conn_weights"]
        self.conn_delays = self.net_dict["conn_delays"]
        self.alpha = self.net_dict["alpha"]
        self.mu_plus = self.net_dict["mu_plus"]
        self.mu_minus = self.net_dict["mu_minus"]
        self.lambda_ = self.net_dict["lambda"]
        self.receptor_type = self.net_dict["receptor_type"]

    def __create_neuronal_populations(self):
        """ Creates the neuronal populations.

        Creates the neuronal populations and stores them as class attributes.

        """
        if nest.Rank() == 0:
            print('Creating neuronal populations.')
        self.spikes_stim_input = nest.Create(
                    'spike_generator', self.n_sensors,
                    params={
                        'spike_times': [],
                        "allow_offgrid_times": True
                    })
        self.pops = []
        for i in range(self.num_pops):
            # Get associated neuron params from the specific population
            neuron_params = self.net_dict["neuron_params"][self.population_names[i]]
            # Create the population
            population = nest.Create(self.net_dict["neuron_model"],
                                    round(self.num_neurons[i]*self.net_dict["N_scaling"]))
            population.set(**neuron_params)
            # print("popppp", nest.GetStatus(population))
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

    def __create_spikes_stim_input(self):
        """ Creates spike generators for external stimulation if specified
        in ``stim_dict``.

        """
        super().__create_spike_stim_input()

    def __connect_neuronal_populations(self):
        """ Creates the connections between neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting neuronal populations recurrently.')
        syn_dict_e = {
            'synapse_model': 'stdp_e',
            'alpha': self.alpha,
            'mu_plus': self.mu_plus,
            'mu_minus': self.mu_minus,
            'lambda': self.lambda_,
            # 'Wmax':1
        }
        syn_dict_i = {
            'synapse_model': 'stdp_i',
            'Wmax' : -0.001,
            'alpha': self.alpha,
            'mu_plus': -1,
            'mu_minus': -1,
            'lambda': self.lambda_,
        }
        
        weight = 'weight'
        #Connect sensors
        prod = self.n_sensors**2
        num_synapses = round((np.log(0.9) / np.log((prod - 1.0) / prod))*self.net_dict['K_scaling']*self.net_dict['N_scaling'])
        conn_dict = {'rule':'fixed_total_number', 'N':num_synapses}
        w = np.random.random(num_synapses)
        syn_dict_e[weight] = w*50
        nest.Connect(self.pops[0][:64], self.pops[0][64:] + self.pops[1], conn_spec= conn_dict, 
                     syn_spec=syn_dict_e)
        # Make excitatory synapses
        prod = self.num_neurons[0]**2
        num_synapses = round((np.log(0.9) / np.log((prod - 1.0) / prod))*self.net_dict['K_scaling']*self.net_dict['N_scaling'])
        conn_dict = {'rule':'fixed_total_number', 'N':num_synapses}
        print(f'N° synapses per neuron: {num_synapses/self.num_neurons[0]}' )
        w = np.random.random(num_synapses)
        syn_dict_e[weight] = w*50
        nest.Connect(self.pops[0][64:], self.pops[0][64:] + self.pops[1], conn_spec= conn_dict,
            syn_spec=syn_dict_e)
        
        # Make inhibitory synapses
        prod = self.num_neurons[1]**2
        num_synapses = round((np.log(0.9) / np.log((prod - 1.0) / prod))*self.net_dict['K_scaling']*self.net_dict['N_scaling'])
        conn_dict = {'rule':'fixed_total_number', 'N':num_synapses}
        print(f'N° synapses per neuron: {num_synapses/self.num_neurons[1]}' )
        w = np.random.random(num_synapses)
        # syn_dict_i[weight] = -w*80
        syn_dict_i[weight] = [-80]*num_synapses
        nest.Connect(self.pops[1], self.pops[0][64:] + self.pops[1], conn_spec= conn_dict,
            syn_spec=syn_dict_i)
        # Connect spike generator to neurons
        # n_p_s = 1 # Neurons per sensors
        # for j, neuron in enumerate(self.pops[0] + self.pops[1]):
        #     if j<self.n_sensors*n_p_s:
        #         nest.Connect(self.spikes_stim_input[j//n_p_s], neuron, syn_spec= {'weight': 545})
        
        # Generate and connect noise signal
        self.noisep = nest.Create('poisson_generator', self.num_neurons[0]+self.num_neurons[1])
        self.noisep.rate = 20
        nest.Connect(self.noisep, self.pops[0]+self.pops[1], syn_spec={"weight": 20.0, "delay":1.5})
        #self.noisem = nest.Create('poisson_generator', self.num_neurons[0]+self.num_neurons[1])
        #self.noisem.rate= 1
        #nest.Connect(self.noisem, self.pops[0]+self.pops[1], syn_spec={"weight": -15.0, "delay":1.5})



        # print(nest.GetConnections(synapse_model = syn_dict['synapse_model']).get('w'))
        print('weights updated')
        

    def __connect_recording_devices(self):
        """ Connects the recording devices to the microcircuit."""
        super().__connect_recording_devices()

    def __connect_poisson_bg_input(self):
        """ Connects the Poisson generators to the microcircuit."""
        super().__connect_poisson_bg_input()

    def __connect_thalamic_stim_input(self):
        """ Connects the thalamic input to the neuronal populations."""
        super().__connect_thalamic_stim_input()


    def __connect_dc_stim_input(self):
        """ Connects the DC generators to the neuronal populations."""
        super().__connect_dc_stim_input()
        
    def __connect_spikes_stim_input(self):
        """ Connects the spikes input to the neuronal populations."""
        super().__connect_spikes_stim_input()