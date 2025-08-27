import nest 
import numpy as np 

from utils.nest_utils import setup_nest

sim_dict = {
    "t_sim": 50.0,
    "sim_resolution": 0.001,
    "rng_seed": 66,
    "local_num_threads": 1,
    "overwrite_files": True,
    "print_time": False,
}

net_dict = {
    "neuron_model": "edlif_psc_alpha_percent0_nestml__with_ed_stdp0_nestml",
    "full_num_neurons": [1, 1],
    "neuron_params": {
        "pop1": {
            "ATP": 100,
            "fix_atp": 0,
            "tau_m": 10,
            "tau_syn_ex": 30,
            "C_m": 220.0,
            "t_ref": 1.0,
            "V_th": 1.0,
        },
        "pop2": {
            "ATP": 100,
            "fix_atp": 0,
            "tau_m": 10,
            "tau_syn_ex": 30,
            "C_m": 220.0,
            "t_ref": 1.0,
            "V_th": 1.0,
        }
    },
    "synapse_model_lib": "ed_stdp0_nestml__with_edlif_psc_alpha_percent0_nestml",
    "synapse_model": "ed_stdp"
}

# Nest Setup
setup_nest(sim_dict)

# Weight recorder
wr = nest.Create('weight_recorder')
nest.CopyModel(net_dict["synapse_model_lib"], net_dict["synapse_model"],
            {
                "weight_recorder": wr,
            })
            # "w": 1.0,
            # "delay": 1.0,
            # "d": 1.0,
            # "receptor_type": 0})
        
# Network
num_neurons = net_dict["full_num_neurons"]
# Create
# Population 1
neuron_params1 = net_dict["neuron_params"]["pop1"]
pop1 = nest.Create(
    net_dict["neuron_model"],
    num_neurons[0])
pop1.set(**neuron_params1)
# Population 2
neuron_params2 = net_dict["neuron_params"]["pop2"]
pop2 = nest.Create(
    net_dict["neuron_model"],
    num_neurons[1])
pop2.set(**neuron_params2)
# Spike recorders
spike_recorders1 = nest.Create("spike_recorder")
spike_recorders2 = nest.Create("spike_recorder")
# Connect
# Connection between populations
syn_dict = {
    'synapse_model': net_dict["synapse_model"],
    #'alpha': 0.5,
    #'mu_plus': 0,
    #'mu_minus': 0,
    #'lambda': 1E-6,
}
nest.Connect(pop1, pop2, 
    syn_spec=syn_dict,
    # conn_spec={
    #     'rule': 'all_to_all',
    # }
)
# Connect spike recorders to populations
nest.Connect(pop1, spike_recorders1)
nest.Connect(pop2, spike_recorders2)

# Nest Prepare and Cleanup
nest.set_verbosity("M_ERROR")
nest.Prepare()
nest.Cleanup()

# Simulation parameters
epochs = 10
firing_rate = 5
simulation_time = sim_dict["t_sim"]

for epoch in range(epochs):
    print("epoch", epoch)
    # Generate spike times
    spike_times = np.linspace(0, simulation_time, int(simulation_time * firing_rate)) #+ net.actual_simulation_time
    #spike_times = [0]
    #print("Spikes", spike_times)
    # External stimulation
    spikes_stim_input = nest.Create(
        'pulsepacket_generator', 
        n=num_neurons[0],
        params={
            'pulse_times': spike_times,
            'activity': int(1E6),
            'sdev': 0.001,
        })
    # spikes_stim_input = nest.Create(
    #     "spike_generator",
    #     params={
    #         "spike_times": spike_times
    #     }
    # )
    #nest.Connect(spikes_stim_input, pop1)
    nest.Connect(spikes_stim_input, pop1, "one_to_one", syn_spec={"delay": 1.})
    # Get connections pre simulation
    syn = nest.GetConnections(
        source=pop1, 
        synapse_model=net_dict["synapse_model"])
    #print("syn", nest.GetStatus(syn))
    initial_weight = nest.GetStatus(syn)[0]["w"]
    # Simulate
    nest.Simulate(sim_dict["t_sim"])

    # Get connections post simulation
    updated_weight = nest.GetStatus(syn)[0]["w"]
    # Plot data
    data1 = nest.GetStatus(spike_recorders1)[0]
    data2 = nest.GetStatus(spike_recorders2)[0]
    # Print data
    # print("Population 1")
    # print(data1)
    # print("Population 2")
    # print(data2)

    #print("Connections")
    #conn = nest.GetConnections()
    #print(conn.get())

    print("initial_weight", initial_weight)
    print("updated_weight", updated_weight)

    # Print params
    #print("Params")
    #print(pop2.get())
