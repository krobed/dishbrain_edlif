# Dishbrain EDLIF

Code used for simulations based in [#Connecting-Neurons-to-a-Mobile-Robot:-An-In-Vitro-Bidirectional-Neural-Interface](https://onlinelibrary.wiley.com/doi/abs/10.1155/2007/12725) article.

# Installation

1. Install NEST, with EDLIF and EDSTDP models. Instructions are found at [edsnn](https://github.com/Wiss/edsnn) repository.

2. Go to `/edsnn` folder: `cd edsnn` and clone this repo.

``` shell
git clone git@github.com:cjotade/dishbrain_edlif.git
```

Go to `dishbrain` folder: `edsnn $ cd dishbrain_edlif/dishbrain`

3. Install requirements

``` shell
pip install -r requirements.txt
```

4. Run the script `run_edsnn_model.py` 

``` shell
dishbrain $ python run_edsnn_model.py
```

Use parse arguments:

`--neuron_model` can be `iaf` (without energy constraints) or `edsnn` (with energy constraints). (default=`edsnn`)

`--neuron_order` dictates the number of inhibitory neurons. Excitatory-Inhibitory ratio is 4:1. (default=20) 

`--atp` dictates the starting value of ATP for both excitatory and inhibitory neurons. Goes from 0 to 100 (default=100). 

`--epochs` dictates the total number of epochs. (default=2500)

5. Every result will be saved at `results/sim_results`. Results include .csv datasets and figures.



