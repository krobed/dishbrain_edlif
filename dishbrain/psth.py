import numpy as np
import matplotlib.pyplot as plt
import nest
import os

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math


def plot_voltages(net):

    path = fr"{os.path.dirname(os.path.abspath(__file__))}/results/sim_results/{sum(net.num_neurons)}/"
    plt.figure(figsize=(10,4))
    ss_times = nest.GetStatus(net.spike_recorders, 'events')
    s_times = {'Spikes': [ss['times'] for ss in ss_times]}
    stop_time = 500
    for multimeter in net.voltmeters[:2]:
        events = nest.GetStatus(multimeter,'events')[0]
        V_m = events['V_m']
        times = events['times']
        senders = events['senders']
        time_mask = times <= stop_time
        times = times[time_mask]
        senders = senders[time_mask]
        V_m = V_m[time_mask]
        # Sort per neuron
        unique_neurons = np.unique(senders)
        for n in unique_neurons:
            mask = senders == n
            plt.plot(times[mask], V_m[mask])
        
            
    for s in range(len(s_times['Spikes'])):
        spikes = s_times['Spikes'][s]
        spikes = spikes[spikes<=stop_time]
        plt.plot(spikes, -67 * np.ones(len(spikes)), "ok", label = 'spikes' if s==0 else None)
    plt.grid()
    plt.title("Voltages of neurons")
    plt.ylabel("V_m [mV]")
    plt.xlabel('Time (ms)')
    plt.savefig(path+f'electrodes/InitVoltagesTrain.png')
    plt.clf()
        # plt.show()


def pick_2x2_by_correlation(psth_counts,
                            top_k_candidates=64,
                            return_score=False,
                            verbose=True):
    """
    Choose 2 input channels and 2 output channels such that:
      input1 correlates highly with output1 and low with output2,
      input2 correlates highly with output2 and low with output1.

    Parameters
    ----------
    psth_counts : ndarray (n_channels, n_bins)
        Binned spike counts (PSTH) for each electrode.
    top_k_candidates : int
        If n_channels is large, restrict search to top_k_candidates channels
        by PSTH std (activity). Increasing gives better search but slower.
    return_score : bool
        If True, return (best_tuple, best_score).
    verbose : bool
        Print picked indices and correlation values.

    Returns
    -------
    best_choice : dict
        {
          'in1': idx, 'in2': idx, 'out1': idx, 'out2': idx,
          'corr_matrix': corr_matrix (n x n),
          'score': best_score
        }
    """
    n_ch = psth_counts.shape[0]
    if n_ch < 4:
        raise ValueError("Need at least 4 channels.")

    # Replace rows with all zeros (no spikes) allowed; keep them but they'll have low std.
    # Compute correlation matrix (channels x channels)
    # If constant rows exist, corrcoef produces NaN; convert NaN -> 0
    with np.errstate(invalid='ignore'):
        corr = np.corrcoef(psth_counts)
    # NaNs -> 0 (no correlation information)
    corr = np.nan_to_num(corr, nan=0.0)

    # Candidate selection: choose channels with largest STD (most informative)
    stds = psth_counts.std(axis=1)
    candidate_idx = np.argsort(stds)[::-1][:min(top_k_candidates, n_ch)]
    if verbose:
        print(f"Using top {len(candidate_idx)} candidate channels by std (most active):")
        print(candidate_idx)

    best_score = -math.inf
    best_choice = None

    # iterate all choices: choose 2 inputs and 2 outputs from candidate set,
    # ensure all four indices are distinct
    # We will iterate over combinations for inputs and combos for outputs (disjoint)
    cand_set = list(candidate_idx)
    for in_pair in combinations(cand_set, 2):
        # allow ordered matching of inputs->outputs (we'll test both possible matchings)
        remaining_for_outputs = [c for c in cand_set if c not in in_pair]
        # if not enough remaining candidates, expand outputs from whole set excluding inputs
        if len(remaining_for_outputs) < 2:
            remaining_for_outputs = [c for c in range(n_ch) if c not in in_pair]

        for out_pair in combinations(remaining_for_outputs, 2):
            i1, i2 = in_pair
            o1, o2 = out_pair

            # compute score for two possible labelings: (i1->o1, i2->o2) and swapped outputs
            s_a = corr[i1, o1] + corr[i2, o2] - corr[i1, o2] - corr[i2, o1]
            s_b = corr[i1, o2] + corr[i2, o1] - corr[i1, o1] - corr[i2, o2]

            # keep the better labeling and remember which mapping is used
            if s_a >= s_b:
                score = s_a
                mapping = (i1, i2, o1, o2)  # in1,in2,out1,out2
            else:
                score = s_b
                mapping = (i1, i2, o2, o1)  # swap outputs

            if score > best_score:
                best_score = score
                best_choice = mapping

    if best_choice is None:
        raise RuntimeError("No valid quadruplet found. Increase top_k_candidates?")

    in1, in2, out1, out2 = best_choice
    if verbose:
        print("Best choice (in1, in2, out1, out2):", best_choice)
        print("Corresponding correlations:")
        print(f"corr(in1,out1) = {corr[in1,out1]:.3f}")
        print(f"corr(in1,out2) = {corr[in1,out2]:.3f}")
        print(f"corr(in2,out1) = {corr[in2,out1]:.3f}")
        print(f"corr(in2,out2) = {corr[in2,out2]:.3f}")
        print("score =", best_score)

    result = {
        'in1': in1, 'in2': in2, 'out1': out1, 'out2': out2,
        'corr_matrix': corr,
        'score': best_score
    }
    return result

def plot_selected_psths(psth_counts, selection, bin_edges=None, figsize=(10,6)):
    """
    Plot PSTHs (bar plots) for the selected four electrodes for quick inspection.
    selection: dict returned by pick_2x2_by_correlation
    psth_counts: (n_channels, n_bins)
    """
    in1, in2, out1, out2 = selection['in1'], selection['in2'], selection['out1'], selection['out2']
    chosen = [in1, out1, in2, out2]  # plot matched pairs next to each other
    labels = [f"in1 ({in1})", f"out1 ({out1})", f"in2 ({in2})", f"out2 ({out2})"]
    n_bins = psth_counts.shape[1]

    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
    x = np.arange(n_bins)
    for ax, ch, lab in zip(axs, chosen, labels):
        ax.bar(x, psth_counts[ch], width=1.0)
        ax.set_ylabel(lab)
    axs[-1].set_xlabel("Bin index")
    plt.tight_layout()
    plt.show()

# -------------------------
# Example usage:
# -------------------------
# assume `psth_counts` is a numpy array of shape (64, n_bins)
# selection = pick_2x2_by_correlation(psth_counts, top_k_candidates=16, verbose=True)
# plot_selected_psths(psth_counts, selection)


def get_electrodes_from_psth(net):
    # --- Extract spike times ---
    spike_trains = []
    path = fr"{os.path.dirname(os.path.abspath(__file__))}/results/sim_results/{sum(net.num_neurons)}/"
    if not os.path.exists(path+'electrodes/'): os.makedirs(path+'electrodes/')
    for det in net.spike_recorders:
        events = nest.GetStatus(det)[0]
        spike_trains.append(events["events"]["times"])

    # --- Make PSTHs ---
    bin_size = 4.0  # ms
    t_start, t_end = 0.0, net.actual_simulation_time
    bins = np.arange(t_start, t_end + bin_size, bin_size)

    psth_counts = []
    for spikes in spike_trains:
        counts, _ = np.histogram(spikes, bins=bins)
        psth_counts.append(counts)

    psth_counts = np.array(psth_counts)  # shape: (n_electrodes, n_bins)

    # --- Heatmap plot ---
    plt.figure(figsize=(12, 8))
    plt.imshow(psth_counts, aspect='auto', cmap="viridis",
            extent=[t_start, t_end, len(net.electrodes), 1])
    plt.colorbar(label='Spike count')
    plt.xlabel('Time (ms)')
    plt.ylabel('Electrode #')
    plt.title('PSTH Heatmap (All Electrodes)')
    plt.savefig(path+'electrodes/psth.png')
    plt.clf()
    # --- Correlation between electrodes ---
    corr_matrix = np.corrcoef(psth_counts)
    print("Correlation matrix between electrodes:")
    print(corr_matrix)

    # Optional: plot correlation matrix
    plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Pearson correlation")
    plt.title("Electrode response correlation")
    plt.xticks(range(len(net.electrodes)), range(1, len(net.electrodes)+1))
    plt.yticks(range(len(net.electrodes)), range(1, len(net.electrodes)+1))
    plt.savefig(path+'electrodes/correlation.png')
    return pick_2x2_by_correlation(psth_counts)