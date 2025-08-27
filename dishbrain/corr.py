import nest
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os

def analyze_network(net, bin_size=5, corr_window_early=(0, 100), corr_window_late=(100, 300)):
    """
    net: dict {electrode_id: spike_recorder}
    bin_size: histogram bin size in ms
    """
    path = fr"{os.path.dirname(os.path.abspath(__file__))}/results/sim_results/{sum(net.num_neurons)}/"
    # --- 1. Extract spikes ---
    spikes = []
    for elec, rec in enumerate(net.spike_recorders):
        events = nest.GetStatus(rec, 'events')[0]
        times = events['times']
        for t in times:
            spikes.append((elec, t))
    spikes = np.array(spikes, dtype=[('elec', int), ('time', float)])
    electrodes = list(range(len(net.electrodes)))

    # --- 2. Define stimulation times (25ms stim + 400ms rest × 5) ---
    stim_times = np.array([i * 450 for i in range(round(net.actual_simulation_time/425))])  # 0, 425, 850, 1275, 1700 ms

    # --- 3. Helper: compute stim-locked PSTH counts ---
    def compute_stim_locked_counts(spikes, electrode, stim_times, window, bin_size):
        t0, t1 = window
        bins = np.arange(t0, t1 + bin_size, bin_size)
        all_counts = []
        for stim in stim_times:
            rel_times = [t - stim for (e, t) in spikes if e == electrode and t0 <= (t - stim) <= t1]
            hist, _ = np.histogram(rel_times, bins=bins)
            all_counts.append(hist)
        return np.sum(all_counts, axis=0), bins

    # --- 4. Correlation between electrodes ---
    def compute_stim_locked_corr(spikes, electrodes, stim_times, window, bin_size):
        counts = {}
        for e in electrodes:
            counts[e], bins = compute_stim_locked_counts(spikes, e, stim_times, window, bin_size)
        mat = np.zeros((len(electrodes), len(electrodes)))
        for i, e1 in enumerate(electrodes):
            for j, e2 in enumerate(electrodes):
                if i == j:
                    mat[i, j] = 1.0
                else:
                    if np.any(counts[e1]) and np.any(counts[e2]):
                        mat[i, j] = np.corrcoef(counts[e1], counts[e2])[0, 1]
                    else:
                        mat[i, j] = np.nan
        return mat, bins

    # --- 5. Early and late correlations ---
    early_mat, bins = compute_stim_locked_corr(spikes, electrodes, stim_times, corr_window_early, bin_size)
    late_mat, _    = compute_stim_locked_corr(spikes, electrodes, stim_times, corr_window_late, bin_size)

    # --- 6. Raster plot ---
    plt.figure(figsize=(10, 6))
    
    for elec in electrodes:
        rel_times = [t for (e, t) in spikes if e == elec and t<=2250]
        plt.scatter(rel_times, [elec] * len(rel_times), s=5)
    for time in stim_times[stim_times<2250]:
        plt.axvline(time, color="red", linestyle="--", label='Stimulation' if time==0.0 else None)
        plt.axvline(time+100, color="blue", linestyle="--", label='Early' if time==0.0 else None)
        plt.axvline(time+300, color="green", linestyle="--", label='Late' if time==0.0 else None)
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Electrode")
    plt.title("Raster plot")
    plt.savefig(path+'/raster.png')

    # --- 7. PSTHs per electrode ---
    fig, axes = plt.subplots(len(electrodes), 1, figsize=(8, 2 * len(electrodes)), sharex=True)
    if len(electrodes) == 1:
        axes = [axes]
    for ax, e in zip(axes, electrodes):
        counts, bins = compute_stim_locked_counts(spikes, e, stim_times, (0, 300), bin_size)
        ax.bar(bins[:-1], counts, width=bin_size, align='edge')
        ax.axvline(0, color="red", linestyle="--")
        ax.set_ylabel(f"E{e}")
    axes[-1].set_xlabel("Time relative to stim (ms)")
    plt.suptitle("Stimulus-locked PSTHs")
    plt.tight_layout()
    fig.savefig(path+'/psths.png')

   

    # --- 9. Find best electrode pairs ---
    best_score = -np.inf
    best_pair = None

    for inp1, inp2 in itertools.combinations(range(len(electrodes)), 2):
        for out1, out2 in itertools.combinations([i for i in range(len(electrodes)) if i not in (inp1, inp2)], 2):
            
            # Correlation constraints (use late response)
            c11 = late_mat[inp1, out1]
            c12 = late_mat[inp1, out2]
            c21 = late_mat[inp2, out1]
            c22 = late_mat[inp2, out2]

            # Score = want high c11 + c22, low c12 + c21
            if not np.isnan([c11, c12, c21, c22]).any():
                score = (c11 + c22) - (c12 + c21)
                if score > best_score:
                    best_score = score
                    best_pair = {'in1': electrodes[inp1], 'out1':electrodes[out2], 'in2':
                                 electrodes[inp2], 'out2':electrodes[out1]}
     # --- 8. Correlation matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    im0 = axes[0].imshow(early_mat, vmin=-1, vmax=1, cmap="bwr")
    axes[0].set_title("Early (0–50 ms)")
    axes[0].set_xticks(range(len(electrodes)))
    axes[0].set_yticks(range(len(electrodes)))
    axes[0].set_xticklabels(electrodes)
    axes[0].set_yticklabels(electrodes)

    im1 = axes[1].imshow(late_mat, vmin=-1, vmax=1, cmap="bwr")
    axes[1].set_title("Late (100–250 ms)")
    axes[1].set_xticks(range(len(electrodes)))
    axes[1].set_yticks(range(len(electrodes)))
    axes[1].set_xticklabels(electrodes)
    axes[1].set_yticklabels(electrodes)

    fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.7, label="Correlation")
    plt.suptitle("Electrode Correlation Matrices")
    fig.savefig(path+'/corr.png')
    return early_mat, late_mat, best_pair, best_score
