import os
import numpy as np
import nest
import warnings
import pandas as pd
from . import network
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
import corr
from matplotlib.patches import ConnectionPatch


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def neurons_near(pos_dict, center, radius):
    return [nid for nid, pos in pos_dict.items() if euclidean(pos, center) <= radius]

def plot_circuits(positions, edges, sources, targets, path):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Opcional: asignar posiciones al grafo
    nx.set_node_attributes(G, positions, 'pos')
    print('Searching for circuits')
    # Encuentra todos los caminos desde pop1 → pop2 (circuitos)
    all_paths = []
    with alive_bar(len(sources)*len(targets)) as bar:
        for source in sources:
            for target in targets:
                paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=2))  # permite hasta 4 saltos
                all_paths.extend(paths)
                bar()
    print(f"Found {len(all_paths)} circuits between input and output.")
    plt.figure(figsize=(10, 6))
    plt.clf()
    # Dibuja el grafo entero
    nx.draw(G, pos=positions, node_size=0.6, node_color='lightgray', arrows=True, edge_color='gray')
    print('Plotting paths')
    # Resalta los caminos encontrados
    with alive_bar(len(all_paths)) as bar:
        for p in all_paths:
            path_edges = list(zip(p[:-1], p[1:]))
            nx.draw_networkx_edges(G, pos=positions, edgelist=path_edges, edge_color='red', width=2)
            bar()
    print('Saving figure')
    plt.title("Circuitos sinápticos (directos e indirectos)")
    plt.axis('off')
    plt.draw()
    plt.savefig(path)




def plot_neuron_positions(pos_dict, exc_neurons, inh_neurons, electrode_pos, r,path, radius=None, centers=None):
    """
    Plots neuron positions.

    Parameters:
    - pos_dict: dict of neuron ID -> [x, y]
    - exc_neurons: NodeCollection of excitatory neurons
    - inh_neurons: NodeCollection of inhibitory neurons
    - radius: optional float, draw a circle of this radius around centers
    - centers: optional list of (x, y) center points for circles
    """

    exc_ids = exc_neurons.tolist()
    inh_ids = inh_neurons.tolist()

    exc_pos = np.array([pos_dict[nid] for nid in exc_ids])
    inh_pos = np.array([pos_dict[nid] for nid in inh_ids])

    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    ax[0].scatter(exc_pos[:, 0], exc_pos[:, 1], c='r', label='Excitatory', alpha=0.6, s=1)
    ax[0].scatter(inh_pos[:, 0], inh_pos[:, 1], c='b', label='Inhibitory', alpha=0.6, s=1)
    ax[1].scatter(exc_pos[:, 0], exc_pos[:, 1], c='r', label='Excitatory', alpha=0.6, s=1)
    ax[1].scatter(inh_pos[:, 0], inh_pos[:, 1], c='b', label='Inhibitory', alpha=0.6, s=1)

    if radius is not None and centers is not None:
        for cx, cy in centers:
            circle = plt.Circle((cx, cy), radius, color='gray', fill=False, linestyle='--')
            ax[0].add_patch(circle)
            circle = plt.Circle((cx, cy), radius, color='gray', fill=False, linestyle='--')
            ax[1].add_patch(circle)
    
    for i,point in enumerate(electrode_pos):
        circle = plt.Circle(point, r, color='black', fill=False)
        ax[0].add_patch(circle)
        circle = plt.Circle(point, r, color='black', fill=False)
        ax[1].add_patch(circle)
        txt_point = [point[0] + r*1.5, point[1] + r*1.5]
        ax[1].text(*txt_point, f'{i}', horizontalalignment='center', verticalalignment='center')

    height= np.max(np.array(electrode_pos).flatten())-np.min(np.array(electrode_pos).flatten()) + 4*r
    width= np.max(np.array(electrode_pos).flatten())-np.min(np.array(electrode_pos).flatten()) + 4*r
    start = np.min(np.array(electrode_pos).flatten()) - 2*r,np.min(np.array(electrode_pos).flatten()) - 2*r
    min_x,max_x = start[0], start[0] + width
    min_y,max_y = start[1],start[1] + height
   
    rectangle = plt.Rectangle(start, width, height, fill=False, ec='black')
    ax[0].add_patch(rectangle)
    xyA_min = [max_x, min_y]
    xyB_min = [min_x, min_y]
    xyA_max = [max_x, max_y]
    xyB_max = [min_x, max_y]
    ax[0].legend()    
    
    con1 = ConnectionPatch(xyA=xyA_min, coordsA=ax[0].transData, 
                        xyB=xyB_min, coordsB=ax[1].transData, color = 'black')
    # Add bottom side to the figures
    fig.add_artist(con1)
    con2 = ConnectionPatch(xyA=xyA_max, coordsA=ax[0].transData, 
                        xyB=xyB_max, coordsB=ax[1].transData, color = 'black')
    # Add upper side to the figure
    fig.add_artist(con2)
    ax[1].set_xlim(min_x, max_x)
    ax[1].set_ylim(min_y, max_y)
    plt.draw()
    fig.savefig(path+'/neurons.png')


def connect_with_gaussian(source_list=None, target_list=None, pos_dict=None, syn_spec=None, sigma=1, weights=None, p_max=0.2):
    if source_list is None: source_list = []
    if target_list is None: target_list = []
    if pos_dict is None: pos_dict = {}
    if syn_spec is None: syn_spec = {}
    if weights is None: 
        weights = {'source': [], 'target': [], 'init_w': []}

    init_w = np.array(weights['init_w'])
    if len(weights['source']) == 0 or len(init_w[init_w < 0]) == 0:
        print('Saving initial weights...')

        # Convert positions to arrays
        src_positions = np.array([pos_dict[src] for src in source_list])
        tgt_positions = np.array([pos_dict[tgt] for tgt in target_list])

        # Compute full distance matrix (shape: [n_src, n_tgt])
        diff = src_positions[:, None, :] - tgt_positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)

        # Gaussian probabilities
        probs = p_max * np.exp(-(dists ** 2) / (2 * sigma ** 2))

        # No self-connections
        src_idx = np.arange(len(source_list))
        tgt_idx = np.arange(len(target_list))
        if np.array_equal(source_list, target_list):
            probs[src_idx, tgt_idx] = 0.0

        # Random mask for connections
        mask = np.random.rand(*probs.shape) < probs

        src_conn, tgt_conn = np.where(mask)

        with alive_bar(len(src_conn)) as bar:
            for s_idx, t_idx in zip(src_conn, tgt_conn):
                bar()

                if 'Wmax' in syn_spec:
                    syn_spec['w'] = np.random.random()*100
                    w_key = 'w'
                else:
                    w_key = 'weight'

                nest.Connect([source_list[s_idx]], [target_list[t_idx]], syn_spec=syn_spec)

                weights['source'].append(source_list[s_idx])
                weights['target'].append(target_list[t_idx])
                weights['init_w'].append(syn_spec[w_key])

        print('Initial weights saved')
    else:
        print('Loading initial weights...')

        # Convert lists to NumPy arrays with explicit types
        sources = np.array(weights['source'], dtype=np.int64)
        targets = np.array(weights['target'], dtype=np.int64)
        init_w  = np.array(weights['init_w'], dtype=np.float64)

        # Masks for excitatory vs inhibitory
        pos_mask = init_w >= 0
        neg_mask = ~pos_mask

        # Excitatory connections (weights come from init_w)
        if np.any(pos_mask):
            exc_sources = sources[pos_mask].tolist()
            exc_targets = targets[pos_mask].tolist()
            exc_weights = init_w[pos_mask].tolist()

            exc_syn_spec = syn_spec.copy()
            exc_syn_spec.update({
                "w": exc_weights
            })

            nest.Connect(
                exc_sources,
                exc_targets,
                conn_spec={"rule": "one_to_one"},
                syn_spec=exc_syn_spec
            )

        # Inhibitory connections (fixed weight)
        if np.any(neg_mask):
            inh_sources = sources[neg_mask].tolist()
            inh_targets = targets[neg_mask].tolist()
            inh_weights = init_w[neg_mask].tolist()
            syn_dict_i = {
                "synapse_model": "static_i",
                "weight": inh_weights
            }

            nest.Connect(
                inh_sources,
                inh_targets,
                conn_spec={"rule": "one_to_one"},
                syn_spec=syn_dict_i
            )

        print('Weights loaded successfully')

    print(f'connections generated')
    
    return weights

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

        # if self.synapse_model is not None:
        #     nest.CopyModel(
        #         self.synapse_model,
        #         "ed_stdp_synapse",
        #         {
        #             # "weight_recorder": wr[0],
        #             # "w": init_w,
        #             # "delay": delay,
        #             # "d": delay,
        #             # "receptor_type": 0
        #         })
        self.path=sim_dict['path']
        self.n_sensors = net_dict['n_sensors']
        self.integration_time = sim_dict['integration_time']
        self.mRFR = None
        self.mLFR = None

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

    def connect_recording(self):
        super().__connect_recording_devices()



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
            spike_times = np.concatenate([spike["events"]["times"] for spike in status])
            return spike_times

        left_sr = self.spike_recorders[self.out1]
        right_sr = self.spike_recorders[self.out2]
        left_spike_times = recorder_to_times(left_sr)
        right_spike_times = recorder_to_times(right_sr)
                
        # print(left_spike_times)
        # print(right_spike_times)
        # Decode actions from spike times
        left_velocity, self.left_fr = self.__decode_action(left_spike_times,self.mLFR)
        right_velocity, self.right_fr = self.__decode_action(right_spike_times,self.mRFR)
        if left_velocity<right_velocity:
            left_velocity = -1
        if left_velocity>right_velocity:
            right_velocity = -1
        return [left_velocity, right_velocity]

    def __decode_action(self, spike_times, mFR):
        """ Decodes the firing rates of the wheel populations to a velocity.
        """
        # Convert spike times to Quantity objects for Elephant
        if len(spike_times) != 0:
            spike_times1 = spike_times[
                spike_times >= (self.actual_simulation_time - self.integration_time)]
            firing_rate= (1000/self.integration_time)*len(spike_times1)
        else:
            firing_rate = 0

        # Exponential function to decode firing rate to velocity
        def firing_rate_to_velocity(firing_rate,mFR):
            if mFR==0 or np.isnan(mFR):
                mFR = 1
            return  np.exp(-(firing_rate/mFR)**2)*3+2 # Velocity according to mean firing rate

        # Decode firing rates to velocities
        wheel_velocity = firing_rate_to_velocity(firing_rate,mFR)

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
        self.eta = self.net_dict["eta"]
        self.receptor_type = self.net_dict["receptor_type"]

    def __create_neuronal_populations(self):
        """ Creates the neuronal populations.

        Creates the neuronal populations and stores them as class attributes.

        """
        if nest.Rank() == 0:
            print('Creating neuronal populations.')
        # self.spikes_stim_input = nest.Create(
        #             'spike_generator', self.n_sensors,
        #             params={
        #                 'spike_times': [],
        #                 "allow_offgrid_times": True
        #             })
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
        p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.path)))))
        self.size = 1.0 #cm
        if os.path.exists(p+f'/{sum(self.num_neurons)}/positions.csv'):
            r = pd.read_csv(p+f'/{sum(self.num_neurons)}/positions.csv')
            self.pos_dict = {int(c):np.array([r[c][0],r[c][1]]) for c in r.columns}
        else:
            # os.makedirs(p+f'/{sum(self.num_neurons)}')
            n = sum(self.num_neurons)
            theta = 2 * np.pi * np.random.rand(n)           # random angle
            r = self.size * np.sqrt(np.random.rand(n))           # random radius (area-uniform)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions = np.column_stack((x, y))
            self.pos_dict = {nid: pos for nid, pos in zip(self.pops[0].tolist()+self.pops[1].tolist(), positions)}
            pd.DataFrame(self.pos_dict).to_csv(p+f'/{sum(self.num_neurons)}/positions.csv', index=False)
        r = 0.01 # 30um
        self.electrode_pos = []
        self.electrodes = []
        for x in np.linspace(-0.25 + r,0.25,8, endpoint=False):
            for y in np.linspace(-0.25 + r,0.25,8, endpoint=False):
                neurons = neurons_near(self.pos_dict, center=[x,y], radius=r)  
                if len(neurons)>0:
                    self.electrodes.append(neurons)
                    self.electrode_pos.append([x,y])

        plot_neuron_positions(self.pos_dict, self.pops[0], self.pops[1], self.electrode_pos, r, p+f'/{sum(self.num_neurons)}')
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
            'synapse_model': 'edstdp_e',
            'alpha': self.alpha,
            'mu_plus': self.mu_plus,
            'mu_minus': self.mu_minus,
            'lambda': self.lambda_,
            'eta': self.eta,
            'Wmax': 100
        }
        syn_dict_i = {
            'synapse_model': 'static_i',
            'weight' : -200
            # 'Wmax':-20,
            # 'alpha': self.alpha,
            # 'mu_plus': 0,
            # 'mu_minus': 0,
            # 'eta': self.eta,
            # 'lambda': self.lambda_,
        }


        '''
        Connections with inhibitory weights
        '''
        path =  os.path.dirname(os.path.dirname(__file__))+'/results/sim_results'
        if os.path.exists(path+f'/{sum(self.num_neurons)}/weights.csv'):
            self.weights = pd.read_csv(path+f'/{sum(self.num_neurons)}/weights.csv')
        else:
            if not os.path.exists(path+f'/{sum(self.num_neurons)}'):
                os.makedirs(path+f'/{sum(self.num_neurons)}')
            self.weights={'source':[],'target':[],'init_w':[]}
        
        if len(self.weights['source'])==0:
            # Make excitatory synapses
            self.weights = connect_with_gaussian(self.pops[0].tolist(), (self.pops[0] + self.pops[1]).tolist(), self.pos_dict,
                        syn_spec=syn_dict_e, sigma=0.5, weights=self.weights)
            
        #     # Make inhibitory synapses
            self.weights = connect_with_gaussian(self.pops[1].tolist(), (self.pops[0] + self.pops[1]).tolist(), self.pos_dict,
                        syn_spec=syn_dict_i, sigma=0.5, weights=self.weights)
            pd.DataFrame(self.weights).to_csv(path+f'/{sum(self.num_neurons)}/weights.csv', index = False)
        
        else:
            connect_with_gaussian(syn_spec=syn_dict_e, weights=self.weights)
            

        # if not os.path.exists(path+ f'/{sum(self.num_neurons)}/inputL_circuits.png'):
        #     all_conns = nest.GetConnections()
        #     edges = list(zip(all_conns.source, all_conns.target))
        #     plot_circuits(self.pos_dict, edges, self.inputL, self.outputL+self.outputR, path+ f'/{sum(self.num_neurons)}/inputL_circuits.png')
        #     plot_circuits(self.pos_dict, edges, self.inputR, self.outputL+self.outputR, path+ f'/{sum(self.num_neurons)}/inputR_circuits.png')

        
        '''
        Noise connections
        '''


        # Generate and connect noise signal
        self.noisep = nest.Create('poisson_generator', sum(self.num_neurons))
        self.noisep.rate = 8
        nest.Connect(self.noisep, self.pops[0]+self.pops[1], conn_spec={'rule':'one_to_one'},syn_spec={"weight": 350})
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

    def get_input_output(self):
        """ Calls external function to get the corresponding electrodes for input and output """
        early_mat, late_mat, self.best_pair, best_score = corr.analyze_network(self)
        self.in1=self.best_pair['in1']
        self.in2=self.best_pair['in2']
        self.out1=self.best_pair['out1']
        self.out2=self.best_pair['out2']
        left_sr = self.spike_recorders[self.out1]
        status = nest.GetStatus(left_sr)
        spike_times = np.concatenate([spike["events"]["times"] for spike in status])
        self.mLFR =  len(spike_times)/(self.actual_simulation_time/1000)
        right_sr = self.spike_recorders[self.out2]
        status = nest.GetStatus(right_sr)
        spike_times = np.concatenate([spike["events"]["times"] for spike in status])
        self.mRFR =  len(spike_times)/(self.actual_simulation_time/1000)