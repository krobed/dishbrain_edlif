import pandas as pd
import matplotlib.pyplot as plt
import os
def plot_voltage(fv,fs, sender_id, tmin=None, tmax=None):
    # Read ASCII file (skip lines starting with '#', split by whitespace)
    df = pd.read_csv(fv, sep=r"\s+", names=['sender','time_ms','V_m'], comment='#')
    sf = pd.read_csv(fs, sep=r"\s+", names=['sender','time_ms','V_m'], comment='#')
    print(df)
    # Filter sender
    df_sender = df[df['sender'] == sender_id]
    sf_sender = sf[sf['sender'] == sender_id]

    if df_sender.empty:
        print(f"No data found for sender {sender_id}")
        return

    # Apply time window if given
    if tmin is not None:
        df_sender = df_sender[df_sender['time_ms'] >= tmin]
        sf_sender = sf_sender[sf_sender['time_ms'] >= tmin]
    if tmax is not None:
        df_sender = df_sender[df_sender['time_ms'] <= tmax]
        sf_sender = sf_sender[sf_sender['time_ms'] <= tmax]

    if df_sender.empty:
        print(f"No data found for sender {sender_id} in time window {tmin}-{tmax} ms")
        return

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(df_sender['time_ms'], df_sender['V_m'], label=f"Sender {sender_id}")
    plt.scatter(sf_sender['time_ms'], [-67]*len(sf_sender['time_ms']), c='black')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title(f"Voltage trace of Sender {sender_id}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_atp(filename, sender_id, tmin=None, tmax=None):
    # Load file, skipping the '#' in the header
    df = pd.read_csv(filename, sep=r"\s+", names=['sender','time_ms','ATP'], comment='#')

    # Filter for the specific sender
    df_sender = df[df['sender'] == sender_id]

    if df_sender.empty:
        print(f"No data found for sender {sender_id}")
        return

    # Apply time window if given
    if tmin is not None:
        df_sender = df_sender[df_sender['time_ms'] >= tmin]
    if tmax is not None:
        df_sender = df_sender[df_sender['time_ms'] <= tmax]

    if df_sender.empty:
        print(f"No data found for sender {sender_id} in time window {tmin}-{tmax} ms")
        return

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(df_sender['time_ms'], df_sender['ATP'], label=f"Sender {sender_id}")
    plt.xlabel("Time (ms)")
    plt.ylabel("ATP")
    plt.title(f"ATP trace of Sender {sender_id}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage: sender 1, between 100 ms and 500 ms
path= '/home/winkrobed/edsnn/dishbrain_edlif/dishbrain/results/sim_results/2500/k/1/66/20250826122125'
time = [i*500 for i in range(600)]
for t in range(len(time)-1):
    plot_voltage(path+'/Vm_all.dat',path+'/Spikes_all.dat' , sender_id=1071, tmin=time[t], tmax=time[t+1])
# Example usage: sender 1, between 100 ms and 500 ms
    plot_atp(path+'/ATP_all.dat', sender_id=1071, tmin=time[t], tmax=time[t+1])
