import os 
import argparse

import nest
from pynestml.frontend.pynestml_frontend import generate_nest_target

NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")


if __name__ == "__main__":
    # Parse the command line arguments
    #parser = argparse.ArgumentParser(description="Generate NESTML code")
    #parser.add_argument("neuron_model", default="edlif_psc_exp_percent", type=str, help="Path to the neuron model")
    #parser.add_argument("models_path", default="./models", type=str, help="Path to the models directory")
    #parser.add_argument("target_path", default="/tmp/nestml-component", type=str, help="Path to the target directory")
    #parser.add_argument("logging_level", default="INFO", type=str, help="Log level")
    #args = parser.parse_args()

    neuron_model = "edlif_psc_exp_percent"
    models_path = "edlif/models"
    target_path = "/tmp/nestml-component"
    logging_level = "INFO"

    # Parameters
    input_path = os.path.join(models_path, f"{neuron_model}.nestml")

    # Generate the NESTML target
    generate_nest_target(
        input_path=input_path, 
        target_path=target_path,
        logging_level=logging_level,
        codegen_opts={
            "nest_path": NEST_SIMULATOR_INSTALL_LOCATION
        })
    # Compile the model
    nest.Install("nestmlmodule")

    # Load the model
    neuron = nest.Create(neuron_model)

    # current generator
    gen = "ac" # dc o ac
    print(f"Using {gen} current generator")


    voltmeter = nest.Create("voltmeter")
    voltmeter.set({"record_from": ["V_m", "ATP", "ATP_c", "ATP_s", "E_ap_tr", "E_ap_c"]}) #, "E_ap_der", "E_ap_tr2"]})
    nest.Connect(voltmeter, neuron)

    if gen == "dc":
        cgs = nest.Create('dc_generator')
        cgs.set({"amplitude": 1.86})
    elif gen == "ac":
        cgs = nest.Create('ac_generator')
        cgs.set({"amplitude": 300.2})
        cgs.set({"offset": 700.2})
        cgs.set({"frequency": 4})
        
    nest.Connect(cgs, neuron)

    sr = nest.Create("spike_recorder")
    nest.Connect(neuron, sr)

    nest.Simulate(500.)

    print("ASDASDADDSD")

    spike_times = nest.GetStatus(sr, keys='events')[0]['times']
    fontsize = 15
    fig, ax = plt.subplots(nrows=5, figsize=(15,15))
    ax[0].plot(voltmeter.get("events")["times"], voltmeter.get("events")["V_m"])
    ax[1].plot(voltmeter.get("events")["times"], voltmeter.get("events")["ATP"])
    ax[1].axhline(y=neuron.get("ATP_h"), c="gray", ls="--")
    ax[2].plot(voltmeter.get("events")["times"], voltmeter.get("events")["ATP_c"], c="r", marker ='.')
    ax[2].axhline(y=neuron.get("ATP_basal"), c="gray", ls="--")
    ax[3].plot(voltmeter.get("events")["times"], voltmeter.get("events")["ATP_s"], c="g")
    ax[4].plot(voltmeter.get("events")["times"], voltmeter.get("events")["E_ap_tr"], c="b", label="AP tr")
    ax[4].plot(voltmeter.get("events")["times"], voltmeter.get("events")["E_ap_c"], c="r", label="AP c")
    #ax[5].plot(voltmeter.get("events")["times"], voltmeter.get("events")["E_ap_tr2"], c="b")
    #ax[5].plot(voltmeter.get("events")["times"], voltmeter.get("events")["E_ap_der"], c="r")
    ax[0].scatter(spike_times, -50 * np.ones_like(spike_times), marker="d", c="orange", alpha=.8, zorder=99)
    for _ax in ax:
        _ax.grid(True)
    ax[0].set_ylabel("v [mV]", fontsize=fontsize)
    ax[1].set_ylabel("ATP", fontsize=fontsize)
    ax[2].set_ylabel("ATP_c", fontsize=fontsize)
    ax[3].set_ylabel("ATP_s", fontsize=fontsize)
    ax[4].set_ylabel("E_ap_tr", fontsize=fontsize)
    ax[4].set_ylabel("E_ap_c", fontsize=fontsize)
    ax[-1].set_xlabel("Time [ms]", fontsize=fontsize)
    ax[4].legend()
    fig.show()