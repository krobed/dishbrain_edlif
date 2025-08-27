# -*- coding: utf-8 -*-
#
# helpers.py
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

"""PyNEST Microcircuit: Helper Functions
-------------------------------------------

Helper functions for network construction, simulation and evaluation of the
microcircuit.

"""

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import os
import nest
import numpy as np
import pandas as pd
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')


def num_synapses_from_conn_probs(conn_probs, popsize1, popsize2):
    """Computes the total number of synapses between two populations from
    connection probabilities.

    Here it is irrelevant which population is source and which target.

    Parameters
    ----------
    conn_probs
        Matrix of connection probabilities.
    popsize1
        Size of first poulation.
    popsize2
        Size of second population.

    Returns
    -------
    num_synapses
        Matrix of synapse numbers.
    """
    prod = np.outer(popsize1, popsize2)
    num_synapses = np.log(1. - conn_probs) / np.log((prod - 1.) / prod)
    return num_synapses


def postsynaptic_potential_to_current(C_m, tau_m, tau_syn):
    r""" Computes a factor to convert postsynaptic potentials to currents.

    The time course of the postsynaptic potential ``v`` is computed as
    :math: `v(t)=(i*h)(t)`
    with the exponential postsynaptic current
    :math:`i(t)=J\mathrm{e}^{-t/\tau_\mathrm{syn}}\Theta (t)`,
    the voltage impulse response
    :math:`h(t)=\frac{1}{\tau_\mathrm{m}}\mathrm{e}^{-t/\tau_\mathrm{m}}\Theta (t)`,
    and
    :math:`\Theta(t)=1` if :math:`t\geq 0` and zero otherwise.

    The ``PSP`` is considered as the maximum of ``v``, i.e., it is
    computed by setting the derivative of ``v(t)`` to zero.
    The expression for the time point at which ``v`` reaches its maximum
    can be found in Eq. 5 of [1]_.

    The amplitude of the postsynaptic current ``J`` corresponds to the
    synaptic weight ``PSC``.

    References
    ----------
    .. [1] Hanuschkin A, Kunkel S, Helias M, Morrison A and Diesmann M (2010)
           A general and efficient method for incorporating precise spike times
           in globally time-driven simulations.
           Front. Neuroinform. 4:113.
           DOI: `10.3389/fninf.2010.00113 <https://doi.org/10.3389/fninf.2010.00113>`__.

    Parameters
    ----------
    C_m
        Membrane capacitance (in pF).
    tau_m
        Membrane time constant (in ms).
    tau_syn
        Synaptic time constant (in ms).

    Returns
    -------
    PSC_over_PSP
        Conversion factor to be multiplied to a `PSP` (in mV) to obtain a `PSC`
        (in pA).

    """
    sub = 1. / (tau_syn - tau_m)
    pre = tau_m * tau_syn / C_m * sub
    frac = (tau_m / tau_syn) ** sub

    PSC_over_PSP = 1. / (pre * (frac**tau_m - frac**tau_syn))
    return PSC_over_PSP


def dc_input_compensating_poisson(bg_rate, K_ext, tau_syn, PSC_ext):
    """ Computes DC input if no Poisson input is provided to the microcircuit.

    Parameters
    ----------
    bg_rate
        Rate of external Poisson generators (in spikes/s).
    K_ext
        External indegrees.
    tau_syn
        Synaptic time constant (in ms).
    PSC_ext
        Weight of external connections (in pA).

    Returns
    -------
    DC
        DC input (in pA) which compensates lacking Poisson input.
    """
    DC = bg_rate * K_ext * PSC_ext * tau_syn * 0.001
    return DC


def adjust_weights_and_input_to_synapse_scaling(
        full_num_neurons,
        full_num_synapses,
        K_scaling,
        mean_PSC_matrix,
        PSC_ext,
        tau_syn,
        full_mean_rates,
        DC_amp,
        poisson_input,
        bg_rate,
        K_ext):
    """ Adjusts weights and external input to scaling of indegrees.

    The recurrent and external weights are adjusted to the scaling
    of the indegrees. Extra DC input is added to compensate for the
    scaling in order to preserve the mean and variance of the input.

    Parameters
    ----------
    full_num_neurons
        Total numbers of neurons.
    full_num_synapses
        Total numbers of synapses.
    K_scaling
        Scaling factor for indegrees.
    mean_PSC_matrix
        Weight matrix (in pA).
    PSC_ext
        External weight (in pA).
    tau_syn
        Synaptic time constant (in ms).
    full_mean_rates
        Firing rates of the full network (in spikes/s).
    DC_amp
        DC input current (in pA).
    poisson_input
        True if Poisson input is used.
    bg_rate
        Firing rate of Poisson generators (in spikes/s).
    K_ext
        External indegrees.

    Returns
    -------
    PSC_matrix_new
        Adjusted weight matrix (in pA).
    PSC_ext_new
        Adjusted external weight (in pA).
    DC_amp_new
        Adjusted DC input (in pA).

    """
    PSC_matrix_new = mean_PSC_matrix / np.sqrt(K_scaling)
    PSC_ext_new = PSC_ext / np.sqrt(K_scaling)

    # recurrent input of full network
    indegree_matrix = \
        full_num_synapses / full_num_neurons[:, np.newaxis]
    input_rec = np.sum(mean_PSC_matrix * indegree_matrix * full_mean_rates,
                       axis=1)

    DC_amp_new = DC_amp \
        + 0.001 * tau_syn * (1. - np.sqrt(K_scaling)) * input_rec

    if poisson_input:
        input_ext = PSC_ext * K_ext * bg_rate
        DC_amp_new += 0.001 * tau_syn * (1. - np.sqrt(K_scaling)) * input_ext
    return PSC_matrix_new, PSC_ext_new, DC_amp_new


def plot_raster(path, name, begin, end, N_scaling, populations, id_sim=None):
    """ Creates a spike raster plot of the network activity.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start plotting spikes (included).
    end
        Time point (in ms) to stop plotting spikes (included).
    N_scaling
        Scaling factor for number of neurons.
    populations
        List of populations names to be plotted.

    Returns
    -------
    None

    """
    #import math
    fs = 16  # fontsize

    sd_names, node_ids, data = __load_meter_data(path, name, begin, end)

    #n_networks = len(sd_names) // 8
    #n_conections = int((math.factorial(n_networks) / (math.factorial(2) * math.factorial(n_networks - 2))) * 2)

    color_list = np.tile(['#0063B2', '#b015b6'], len(sd_names)//2)

    last_node_id = node_ids[-1, -1]
    mod_node_ids = np.abs(node_ids - last_node_id) + 1

    label_pos = [(mod_node_ids[i, 0] + mod_node_ids[i, 1]) / 2.
                    for i in np.arange(0, len(populations))]

    stp = 1
    if N_scaling > 0.1:
        stp = int(10. * N_scaling)
        print('  Only spikes of neurons in steps of {} are shown.'.format(stp))

    plt.figure(figsize=(6, 16))
    for i, n in enumerate(sd_names):
        times = data[i]['time_ms']
        neurons = np.abs(data[i]['sender'] - last_node_id) + 1
        plt.plot(times[::stp], neurons[::stp], '.', color=color_list[i])
    plt.xlabel('time [ms]', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(label_pos, populations, fontsize=fs)
    plt.title('Spike raster plot', fontsize=22)
    #if id_sim:
    #    plt.title(f"ID: {id_sim}", fontsize=fs)
    #else:
    #    plt.title(f"ID: {name}", fontsize=fs)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'raster_plot_'+format(nest.rng_seed)+'.png'), dpi=300)


def plot_voltages(path, name, begin, end, populations, firing_rates_name=None, input_names=None):
    """ Computes voltages per population.

    The voltage of each neuron in each population is computed and stored
    in a .dat file in the directory of the voltmeters.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start calculating the firing rates (included).
    end
        Time point (in ms) to stop calculating the firing rates (included).
    populations
        List of populations names to be plotted.

    Returns
    -------
    None

    """
    fs = 18  # fontsize

    i_axes = 0
    sd_names, node_ids, data = __load_meter_data(path, name, begin, end)
    if firing_rates_name:
        _, _, data_fr = __load_meter_data(path, firing_rates_name, begin, end) 
    if input_names:
        i_axes += 1
        data_input = {}
        for input_name in input_names:
            _, _, data_in = __load_meter_data(path, input_name, begin, end)
            data_input[input_name] = data_in

    fig, axs = plt.subplots(len(populations)+i_axes, 1, figsize=(8, 6), sharex=True, sharey=False)
    for i, n in enumerate(sd_names):
        times, voltage = data[i]['time_ms'], data[i]['voltage']
        axs[i].plot(times, voltage, label=populations[i], color=f"C{i}")
        if firing_rates_name and len(data_fr[i]) != 0:
            firing_rates = pd.DataFrame(data_fr[i]).values[:, 1]
            axs[i].plot(firing_rates, -50*np.ones(len(firing_rates)), f"k.", label=f"FR {populations[i]}")
        axs[i].grid()
        axs[i].legend()
        axs[i].set_ylim(-66, -49)
        axs[i].set_ylabel('voltage [mV]')
    if input_names:
        for j, input_name in enumerate(input_names):
            for k in range(len(sd_names)):
                if input_name == "dc_input":
                    times, var = data_input[input_name][k]['time_ms'], data_input[input_name][k]['I']
                    axs[i+1].plot(times, var, "r", 
                    label="dc input" if k == 0 else "")
                else:
                    var = pd.DataFrame(data_input[input_name][k]).values[:, 1]
                    axs[i+1].plot(var, -50*np.ones(len(var)), "k.",
                    label="poisson input" if k == 0 else "")
        axs[i+1].grid()
        axs[i+1].legend()
        axs[i+1].set_ylabel('input [pA]')
    fig.supxlabel('time [ms]', fontsize=fs)
    #fig.supylabel('voltage [mV]', fontsize=fs)
    fig.savefig(os.path.join(path, 'voltage_plot.png'), dpi=300)

def plot_network(path, populations, conn_weights, conn_weights_th=None):
    """ Plots the network structure.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    populations
        List of populations names.
    conn_weights
        List of connection weights.
    conn_weights_th
        List of connection weights for the thresholded network.

    Returns
    -------
    None

    """
    import networkx as nx 
  
    # Generate network graph and params
    G = nx.DiGraph(np.array(conn_weights))
    G = nx.relabel_nodes(G, {i: populations[i] for i in range(len(populations))})
    pos = nx.spring_layout(G, k=len(populations))  # For better example looking
    if conn_weights_th is not None:
        G.add_node("TH")
        pos = {**pos, "TH": (0, 0)}
        for i, w in enumerate(conn_weights_th):
            G.add_edge("TH", populations[i], weight=w)
    labels = nx.get_edge_attributes(G, 'weight')
    # Plot the network
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    edge_labels = []
    for u,v,d in G.edges(data=True):
        if u == v:
            edge_labels.append(
                ((u,v,), f'{d["weight"]}\n\n\n')
            )
        elif (labels.get((v,u)) is not None):
            edge_labels.append(
                ((u,v,),f'{d["weight"]}\n\n\n{labels[(v,u)]}')
            )
        else:
            edge_labels.append(
                ((u,v,), d["weight"])
           )
    edge_labels = dict(edge_labels)

    nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.08', arrowsize=15, ax=ax, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)
    ax.set_title('Network structure', fontsize=12)
    fig.savefig(os.path.join(path, 'network_plot.png'), dpi=300)

def firing_rates(path, name, begin, end):
    """ Computes mean and standard deviation of firing rates per population.

    The firing rate of each neuron in each population is computed and stored
    in a .dat file in the directory of the spike recorders. The mean firing
    rate and its standard deviation are printed out for each population.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start calculating the firing rates (included).
    end
        Time point (in ms) to stop calculating the firing rates (included).

    Returns
    -------
    None

    """
    sd_names, node_ids, data = __load_meter_data(path, name, begin, end)
    all_mean_rates = []
    all_std_rates = []
    for i, n in enumerate(sd_names):
        senders = data[i]['sender']
        # 1 more bin than node ids per population
        bins = np.arange(node_ids[i, 0], node_ids[i, 1] + 2)
        spike_count_per_neuron, _ = np.histogram(senders, bins=bins)
        rate_per_neuron = spike_count_per_neuron * 1000. / (end - begin)
        np.savetxt(os.path.join(path, ('rate' + str(i) + '.dat')),
                   rate_per_neuron)
        # zeros are included
        all_mean_rates.append(np.mean(rate_per_neuron))
        all_std_rates.append(np.std(rate_per_neuron))
    print('Mean rates: {} spikes/s'.format(np.around(all_mean_rates, decimals=3)))
    print('Standard deviation of rates: {} spikes/s'.format(
        np.around(all_std_rates, decimals=3)))


def boxplot(path, populations):
    """ Creates a boxblot of the firing rates of all populations.

    To create the boxplot, the firing rates of each neuron in each population
    need to be computed with the function ``firing_rate()``.

    Parameters
    -----------
    path
        Path where the firing rates are stored.
    populations
        Names of neuronal populations.

    Returns
    -------
    None

    """
    fs = 16
    pop_names = [string.replace('23', '2/3') for string in populations]
    label_pos = list(range(len(populations), 0, -1))
    color_list = ['#b015b6','#0063B2']
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')
    meanprops = dict(linestyle='--', linewidth=2.5, color='lightgray')

    rates_per_neuron_rev = []
    for i in np.arange(len(populations))[::-1]:
        rates_per_neuron_rev.append(
            np.loadtxt(os.path.join(path, ('rate' + str(i) + '.dat'))))

    plt.figure(figsize=(6, 4))
    bp = plt.boxplot(rates_per_neuron_rev, 0, 'rs', 0, medianprops=medianprops,
                     meanprops=meanprops, meanline=True, showmeans=True)
    plt.setp(bp['boxes'], color='black', linewidth=2)
    plt.setp(bp['whiskers'], color='black', linewidth=2.5)
    plt.setp(bp['caps'], color='black', linewidth=2)
    plt.setp(bp['fliers'], color='red', marker='*')

    # boxcolors
    for i in np.arange(len(populations)):
        boxX = []
        boxY = []
        box = bp['boxes'][i]
        for j in list(range(5)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=color_list[k])
        plt.gca().add_patch(boxPolygon)
    plt.xlabel('firing rate [spikes/s]', fontsize=fs)
    plt.yticks(label_pos, pop_names, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.title('Firing rates', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'box_plot.png'), dpi=300)


def __gather_metadata(path, name):
    """ Reads names and ids of spike recorders and first and last ids of
    neurons in each population.

    If the simulation was run on several threads or MPI-processes, one name per
    spike recorder per MPI-process/thread is extracted.

    Parameters
    ------------
    path
        Path where the spike recorder files are stored.
    name
        Name of the spike recorder, typically ``spike_recorder``.

    Returns
    -------
    sd_files
        Names of all files written by spike recorders.
    sd_names
        Names of all spike recorders.
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load filenames
    sd_files = []
    sd_names = []
    for fn in sorted(os.listdir(path)):
        if fn.startswith(name):
            sd_files.append(fn)
            # spike recorder name and its ID
            fnsplit = '-'.join(fn.split('-')[:-1])
            if fnsplit not in sd_names:
                sd_names.append(fnsplit)

    # load node IDs
    node_idfile = open(os.path.join(path, 'population_nodeids.dat'), 'r')
    node_ids = []
    for node_id in node_idfile:
        node_ids.append(node_id.split())
    node_ids = np.array(node_ids, dtype='i4')
    return sd_files, sd_names, node_ids

def __load_meter_data(path, name, begin, end):
    """ Loads spikes or voltages from recorders.

    Parameters
    ----------
    path
        Path where the files with the data are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start loading data (included).
    end
        Time point (in ms) to stop loading data (included).

    Returns
    -------
    data
        Dictionary containing data in the interval from ``begin``
        to ``end``.

    """
    sd_files, sd_names, node_ids = __gather_metadata(path, name)
    data = {}
    dtype = {'names': ('sender', 'time_ms'),  # as in header
                'formats': ('i4', 'f8')}
    if any('voltmeter' in sd_name for sd_name in sd_names):
        dtype['names'] = ('sender', 'time_ms', 'voltage')
        dtype['formats'] = ('i4', 'f8', 'f8')
    if any('dc_input' in sd_name for sd_name in sd_names):
        dtype['names'] = ('sender', 'time_ms', 'I')
        dtype['formats'] = ('i4', 'f8', 'f8')
    for i, name in enumerate(sd_names):
        data_i_raw = np.array([[]], dtype=dtype)
        for j, f in enumerate(sd_files):
            if name in f:
                # skip header while loading
                ld = np.loadtxt(os.path.join(path, f), skiprows=3, dtype=dtype)
                data_i_raw = np.append(data_i_raw, ld)

        data_i_raw = np.sort(data_i_raw, order='time_ms')
        # begin and end are included if they exist
        low = np.searchsorted(data_i_raw['time_ms'], v=begin, side='left')
        high = np.searchsorted(data_i_raw['time_ms'], v=end, side='right')
        data[i] = data_i_raw[low:high]
    return sd_names, node_ids, data

def draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default=None)
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
        # Informative exception for multiedges
        try:
            (u, v) = next(iter(labels))  # ensures no edge key provided
        except ValueError as err:
            raise nx.NetworkXError(
                "draw_networkx_edge_labels does not support multiedges."
            ) from err
        except StopIteration:
            pass

    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        if n1 == n2:
            y += 0.075
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items