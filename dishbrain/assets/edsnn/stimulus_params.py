import numpy as np

stim_dict = {
    # turn thalamic input on or off (True or False)
    'thalamic_input': False,
    # start of the thalamic input (in ms)
    'th_start': 0.0,
    # duration of the thalamic input (in ms)
    'th_duration': 0.0,
    # rate of the thalamic input (in spikes/s)
    'th_rate': 20.0,
    # number of thalamic neurons
    'num_th_neurons': 1,
    # connection  of the thalamus to the different populations
    # (same order as in 'populations' in 'net_dict')
    #'conn_weights_th':
    #    np.array([10, 10, 5]),
    'conn_weights_th':
        np.array([1000, 1500, 200]),
    # connection delays of the thalamus to the different populations
    # (same order as in 'populations' in 'net_dict')
    'conn_delays_th': np.array([1, 1, 1]),
    # optional DC input
    # turn DC input on or off (True or False)
    'dc_input': False,
    # start of the DC input (in ms)
    'dc_start': 0.1,
    # duration of the DC input (in ms)
    'dc_dur': 990.0,
    # amplitude of the DC input (in pA); final amplitude is population-specific
    # and will be obtained by multiplication with 'K_ext'
    'dc_amp': 1000000.0,
    # Spike generator parameters
    'spikes_input': False,
}