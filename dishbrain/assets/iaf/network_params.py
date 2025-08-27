import numpy as np

order= 20 # Number of excitatory neurons
# n_inh = round(n_ex/4) # Number of inhibitory neurons (4:1 exc-inh)
net_dict = {
    # factor to scale the number of neurons
    'N_scaling': 1,
    'K_scaling': 1,
    # 'input_scale': 0.1, # percentage of neurons used as input
    # 'output_scale': 0.2, # percentage of neurons used as output (to be divided into left and right output)
    # neuron model
    # 'neuron_model': 'edlif_psc_alpha_percent0_nestml__with_ed_stdp0_nestml',
    'neuron_model': 'iaf_psc_exp',
    # names of the simulated neuronal populations
    # 'populations': ['SI', "H1", "MO1", "MO2"],
    'populations': ['E','I'],
    # number of neurons in the different populations (same order as
    # 'populations')
    'full_num_neurons':
        # [5, 1, 1, 1],
        [order*4, order],
    'neuron_params': {
        'E': {
            "C_m": 250,
            "tau_m": 20,
            "tau_syn_ex": 0.5,
            "tau_syn_in": 0.5,
            "t_ref": 2.0,
            "E_L": -70.0,
            "V_th":-65.0,
            "V_reset": -70.0,
        },
        'I': {
            "C_m": 250,
            "tau_m": 20,
            "tau_syn_ex": 0.5,
            "tau_syn_in": 0.5,
            "t_ref": 2.0,
            "E_L": -70.0,
            "V_th":-65.0,
            "V_reset": -70.0,
        },
    },
    # synaptic connections between the populations
    # "synapse_model_lib": "ed_stdp0_nestml__with_edlif_psc_alpha_percent0_nestml",
    "synapse_model_lib": "stdp_synapse",
    # "synapse_model": "ed_stdp",
    "synapse_model": "stdp",
    #"synapse_model": None,
    # connection weights (the first index corresponds to the targets
    # and the second to the sources)
    "conn_weights": np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]),
    # connection delays (the first index corresponds to the targets
    # and the second to the sources)
    "conn_delays": np.array([
            [1, 1, 1],  
            [1, 1, 1],
            [1, 1, 1]
        ]),
    # STDP parameters
    "alpha": 0.5,
    "mu_plus": 0,
    "mu_minus": 0,
    # "lambda": 1E-2,
    "lambda": 0.001,
    "receptor_type": 0,
    'tau_tr_pre': 0.01, 
    'tau_tr_post': 0.01
}
    