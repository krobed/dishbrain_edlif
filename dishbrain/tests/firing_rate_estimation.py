import numpy as np
import elephant.statistics as es
import quantities as pq
import neo

import matplotlib.pyplot as plt

#simulation_time = 40  # ms
#actual_simulation_time = 40  # ms
simulation_time = 40
actual_simulation_time = 640
fs = 100  # Hz

# Create a spike train with fs
#spike_times = np.linspace(0, simulation_time, simulation_time * fs)
spike_times = np.array([662.038, 667.151, 670.173, 672.386, 674.195, 675.776, 677.217, 678.56, 679.83])


# Calculate inter-spike intervals (ISIs)
isis = np.diff(spike_times)

# Calculate the mean ISI
mean_isi = np.mean(isis)
mean_isi_seconds = mean_isi / 1000
firing_rate_corrected = 1 / mean_isi_seconds
print("SPIKE TIMES", spike_times)
spike_times = neo.SpikeTrain(
    spike_times, 
    units='ms', 
    t_start=actual_simulation_time,
    t_stop=actual_simulation_time+simulation_time
    #t_start=(actual_simulation_time - simulation_time),
    #t_stop=actual_simulation_time)
)
print("SPIKE TIMES neo", spike_times)

# Calculate firing rate using Elephant
firing_rate = es.instantaneous_rate(
    spike_times, 
    sampling_period=0.001 * pq.ms, 
    kernel='auto',
    t_start=(actual_simulation_time) * pq.ms,
    t_stop=(actual_simulation_time+simulation_time) * pq.ms)

# Print the estimated firing rate
#print(np.mean(firing_rate.magnitude))
#print(np.mean(np.array(firing_rate).squeeze())/simulation_time)
print(np.max(firing_rate.magnitude)/1000)
print("FIRING RATE (corrected): ", firing_rate_corrected/1000, "Hz")

# # Plotting the firing rate over time
plt.figure(figsize=(10, 5))
plt.plot(firing_rate.times, firing_rate.magnitude, label='Estimated Firing Rate')
plt.xlabel('Time (ms)')
plt.ylabel('Estimated Firing Rate (Hz)')
plt.title('Estimated Firing Rate Over Time')
plt.grid(True)
plt.show()