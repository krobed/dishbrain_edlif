import nest
import nest.random
import nest.voltage_trace
import matplotlib.pyplot as plt
import numpy as np

module_name = "edlif_psc_alpha_0_module"
nest.Install(module_name)
# Reset NEST kernel
nest.ResetKernel()

sim_time = 2500  # Simulation time in ms
wr_E = nest.Create("weight_recorder")
wr_I = nest.Create("weight_recorder")


# Define the custom STDP model with negative weights
nest.CopyModel("ed_stdp0_nestml__with_edlif_psc_alpha_percent0_nestml", "edstdp_e", 
                  {"weight_recorder": wr_E})
    
nest.CopyModel('static_synapse', "static_i", 
                  {"weight_recorder": wr_I})
    
# Define neuron and synapse parameters
num_neurons = 1000  # Total number of neurons
num_excitatory = int(0.8 * num_neurons)  # 80% excitatory
num_inhibitory = num_neurons - num_excitatory  # 20% inhibitory

neuron_params = {
    # "V_th": 15,
    # "V_reset":5,
    "tau_m": 20.0,  # Membrane time constant
    "t_ref": 8.0,  # Refractory period
    "tau_syn_ex": 0.5,  # Excitatory synaptic time constant
    "tau_syn_in": 0.5,  # Inhibitory synaptic time constant
}

# Create neurons
neurons = nest.Create("edlif_psc_alpha_percent0_nestml__with_ed_stdp0_nestml", num_neurons)
# for n in neurons:
    # n.I_e = np.random.normal(166, 15)

# Synapse parameters
syn_params_excitatory = {
    "synapse_model": "edstdp_e",  # Use custom STDP model
    "Wmax": 1000.0,   # Maximum weight (can be positive or negative)
    "lambda": 0.01,  # STDP update rate
    "eta": 50,
    "alpha": 0.5,    # STDP learning rate
    "mu_plus": 0,   # Potentiation factor (increase weight, even if negative)
    "mu_minus": 0,  # Depression factor (decrease weight, even if negative)
    "w": 200,  # Initial weight (can be positive or negative)
}


syn_params_inhibitory = {
    "synapse_model": "static_i",  # Use custom STDP model
    "weight": -400,  # Static inhibitory weight
}


# Set connection specs
conn_dict = {'rule': 'pairwise_bernoulli', 'p':0.1}
# Connect neurons
# Excitatory connections
nest.Connect(
    neurons[:num_excitatory],  # Source neurons (excitatory)
    neurons,  # Target neurons (all neurons)
    conn_spec=conn_dict,
    syn_spec=syn_params_excitatory,
)
conn_dict = {'rule': 'pairwise_bernoulli', 'p':0.1}
# Inhibitory connections
nest.Connect(
    neurons[num_excitatory:],  # Source neurons (inhibitory)
    neurons,  # Target neurons (all neurons)
    conn_spec=conn_dict,
    syn_spec=syn_params_inhibitory,
)


# Add Poisson noise generator
poisson_noise = nest.Create("poisson_generator", num_neurons-10, params={"rate": 0.5})  # 1000 Hz Poisson noise
nest.Connect(poisson_noise, neurons[10:],conn_spec={"rule":"all_to_all"}, syn_spec= {"weight": 1000})

generator = nest.Create("spike_generator", 1, {"spike_times": [1000], "spike_weights": [1500]})
nest.Connect(generator, neurons[:10], conn_spec= {'rule': 'all_to_all'})

# Add recording devices
spike_detector = nest.Create("spike_recorder")
voltmeter = nest.Create("voltmeter")


# Connect weight recorder to excitatory connections

nest.Connect(neurons, spike_detector)  # Record spikes from the first neuron
nest.Connect(voltmeter, neurons)  # Record membrane potential from the first neuron


# Simulate the network

nest.Simulate(sim_time)

# Plot results
# Membrane potential
plt.figure()
nest.voltage_trace.from_device(voltmeter)
plt.title("Membrane Potential")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")


# Spike raster plot
plt.figure()
spike_times = nest.GetStatus(spike_detector, "events")[0]["times"]
spike_senders = nest.GetStatus(spike_detector, "events")[0]["senders"]
plt.plot(spike_times, spike_senders, ".")
plt.title("Spike Raster Plot")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")


# Plot weight changes
plt.figure()
w_E = nest.GetConnections(synapse_model = 'edstdp_e').get('w')
w_I = nest.GetConnections(synapse_model = 'static_i').get('weight')
# print(w_E)
plt.plot(w_E, ".")
# plt.plot(w_I, ".")
plt.title("Synaptic Weight Changes")
plt.xlabel("Time (ms)")
plt.ylabel("Synaptic Weight")
plt.show()
