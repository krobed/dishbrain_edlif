# %%

import matplotlib.pyplot as plt
import nest
import numpy as np
import re
import os

from pynestml.frontend.pynestml_frontend import generate_nest_target

NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")

# %%
module_name = "edlif_psc_alpha_0_module"
nest.Install(module_name)

# %%
def experiment(epochs:int = 50, 
              seed: int = 1,
              n_pre_neurons: int = 1000,
              n_post_neurons: int= 100,
              neuron_model: str = "edlif_psc_alpha_percent0_nestml__with_ed_stdp0_nestml",
              syn_spec: dict = {"synapse_model": "ed_stdp0_nestml__with_edlif_psc_alpha_percent0_nestml",
                               "alpha": 0.5,
                               "mu_minus": 0,
                               "mu_plus": 0,
                               "lambda": 0.01,
                               "eta": 50},
              sim_time: float = 2000,
              neuron_params: dict = {'tau_syn_ex': 6, 'tau_syn_in': 6, 't_ref':2,'tau_syn_atp_ex': 60, 'tau_syn_atp_in': 60, 'gamma':20}):
    # reset kernel
    nest.ResetKernel()
    # set seeds
    nest.rng_seed = seed
    np.random.seed(seed)
    # create neurons
    
    noisep = nest.Create('poisson_generator', n_pre_neurons+n_post_neurons)
    noisep.rate= 1
    noisem = nest.Create('poisson_generator', n_pre_neurons+n_post_neurons)
    noisem.rate= 1
    neuron_pre = nest.Create(neuron_model, n_pre_neurons)
    neuron_post = nest.Create(neuron_model, n_post_neurons)
    
    for neuron in [neuron_pre, neuron_post]:
        neuron.set(**neuron_params)
        neuron.tau_m = 20 #20
        neuron.V_th = 15
        neuron.V_reset = 10
        neuron.C_m = 250 #250
        # energy params
        neuron.K_ATP = 1
        neuron.E_ap = 4.1
        neuron.E_rp = 5
        neuron.E_hk = 5
        neuron.E_syn_ex = 0.5
        neuron.E_syn_in = 0.5
        
    # **OBS** post synaptic neurons also has his own I_e
    

    # recorders
    sr_post = nest.Create("spike_recorder")
    nest.Connect(neuron_post, sr_post)
    sr_post_post = nest.Create("spike_recorder")
    
    sr_pre = nest.Create("spike_recorder")
    nest.Connect(neuron_pre, sr_pre)
    mult = nest.Create('multimeter',
                      params={"record_from": ['V_m', 'ATP']})
    # nest.Connect(mult, neuron_pre)
    nest.Connect(mult, neuron_post)
    
    nest.Connect(noisep, neuron_pre+ neuron_post, conn_spec = {'rule': 'one_to_one'}, syn_spec={"weight": 40.0, "delay": 1.5})
    nest.Connect(noisem, neuron_pre+ neuron_post, conn_spec = {'rule': 'one_to_one'}, syn_spec={"weight": -40.0, "delay": 1.5})
    wr = nest.Create("weight_recorder")
    nest.CopyModel(syn_spec["synapse_model"], "stdp", 
                  {"weight_recorder": wr})
    # conn_spec = {'rule':'fixed_total_number', 'N':round(0.05*(n_pre_neurons*n_post_neurons))}
    
    # w = np.random.random(n_pre_neurons)*100
    nest.Connect(neuron_pre, neuron_post,# conn_spec= conn_spec,
                 syn_spec={"synapse_model": "stdp", #'w':w
                           })
   
    
    syn = nest.GetConnections(source=neuron_pre,
                         target=neuron_post,
                         synapse_model="stdp")
  
    #print(syn)
    for param, value in syn_spec.items():
        if param != "synapse_model":
            nest.SetStatus(syn, {param: value})

    # actual_sim_time = 0.0
    generator = nest.Create("spike_generator", n_pre_neurons, {"spike_times": [1,1001,2001,3001,4001], "spike_weights": [545]*5})
    nest.Connect(generator, neuron_pre, conn_spec= {'rule': 'one_to_one'})
    nest.Simulate(sim_time)
    # for i in range(epochs):
    #     print(f'Simulation N°: {i+1}')
    #     # generator.spike_times = [actual_sim_time+1]
    #     nest.Simulate(sim_time)
    #     actual_sim_time+=sim_time
    sr_pre = sr_pre.get("events")
    sr_post = sr_post.get("events")
    sr_post_post = sr_post_post.get("events")
    mult = mult.get("events")
    wr = wr.get("events")
    fin_weights = nest.GetConnections(synapse_model="stdp").get('w')
    return sr_pre, sr_post, mult, wr, fin_weights

# %%
def plots(mult, wr, eta, fin_weights, eq_energy_level):

    plt.figure()
    plt.title(fr"$\eta$ = {eta}")
    plt.xlabel("time (ms)")
    plt.ylabel("ATP")
    plt.plot(mult['times'], mult['ATP'])
    plt.axhline(eq_energy_level, c='grey', ls='--')

    plt.figure()
    plt.title(fr"$\eta$ = {eta}")
    plt.xlabel("time (ms)")
    plt.ylabel("Voltage")
    plt.plot(mult['times'], mult['V_m'])

    plt.figure()
    plt.title(fr"$\eta$ = {eta}")
    plt.xlabel("time (ms)")
    plt.ylabel("Weights")
    plt.plot(wr['times'], wr['weights'], '.')

    plt.figure()
    plt.xlabel("Weights")
    plt.ylabel("Frequency")
    plt.title(fr"$\eta$ = {eta}")
    plt.hist(fin_weights)    

# %%
def equall_dep_pot_energy_level(eta: float, alpha: float = 0.5, a_h: float = 100):
    """gives A* that is achieve when depression and 
    potentiation has the same energy level"""
    if eta == 0:
        return 0
    else:
        return a_h*(np.log(alpha)/eta + 1)

# %% [markdown]
# ### Simulating base case that will be compared to the other simulations with the evolutionary algorithm.

# %%
# Base case
eta = 50
n_pre_neurons = 1000
n_post_neurons= 1
alpha = 0.5
mu_plus = 0.0
mu_minus = 0.0

syn_spec = {"synapse_model": "ed_stdp0_nestml__with_edlif_psc_alpha_percent0_nestml",
           "alpha": alpha,
           "mu_minus": mu_minus,
           "mu_plus": mu_plus,
           "lambda": 0.01,
           "eta": eta}

eq_energy_level = equall_dep_pot_energy_level(eta=eta,
                                              alpha=alpha,
                                              a_h=100)

sr_pre, sr_post, mult_base, wr, fin_weights = experiment(n_pre_neurons=n_pre_neurons,
                                       n_post_neurons=n_post_neurons,
                                       syn_spec=syn_spec)


plots(mult=mult_base, 
      wr=wr, 
      eta=eta, 
      fin_weights=fin_weights, 
      eq_energy_level=eq_energy_level)


# %%
defaults =nest.GetDefaults("edlif_psc_alpha_percent0_nestml__with_ed_stdp0_nestml")
print("Available state variables: edlif_psc_alpha_percent0_nestml__with_ed_stdp0_nestml", ":", defaults["recordables"])

defaults =nest.GetDefaults("iaf_psc_exp")
print("Available state variables: iaf_psc_exp", ":", defaults["recordables"])

# %%
from sklearn.metrics import root_mean_squared_error
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# %% [markdown]
# ### Create evaluation function and initialize PSO algorithm with *pymoo*

# %%
class V_ATP_Error():
    def __init__(self,invalid_fitness=9999999):
        self.__invalid_fitness = invalid_fitness
        self.get_data()
    
    def get_data(self):
        self.v_m = mult_base['V_m']
        self.atp = mult_base['ATP']

    def get_error(self, param):
        try:
            n_pre_neurons= 256
            n_post_neurons= 1
            _,_,Y_pred,_,_  = experiment(n_pre_neurons=n_pre_neurons, n_post_neurons=n_post_neurons, neuron_params=param, syn_spec=syn_spec)
            voltage = Y_pred['V_m']
            atp = Y_pred['ATP']
            v_error = root_mean_squared_error(self.v_m, voltage)
            atp_error = root_mean_squared_error(self.atp, atp)
            return v_error, atp_error 
            
        except Exception as e: 
            error= self.__invalid_fitness
        if error==None:
            error = self.__invalid_fitness
        return error
    
    def evaluate(self, param):
        if param is None:
            return self.__invalid_fitness
        fitness_v, fitness_atp = self.get_error(param)
        return np.mean([fitness_atp,fitness_v]), {'Fitness V_m': fitness_v, 'Fitness ATP': fitness_atp}
    
        


# %%
eval_func = V_ATP_Error()

# %%
variables = ['tau_syn_ex', 'tau_syn_in', 't_ref','tau_syn_atp_ex', 'tau_syn_atp_in', 'gamma'] 
class MyProblem(Problem):
    def __init__(self, n_var = 6, xl= [0]*6, xu = [100]*6):
        super().__init__(
            n_var=n_var,  
            n_obj=1,  
            n_constr=0,  
            xl=xl,  
            xu=xu,  
        )

    def _evaluate(self, x, out,*args, **kwargs):
        def eval_ind(c):
            parameters = {}
            for i in range(len(variables)):
                parameters[variables[i]] = c[i]
            return eval_func.evaluate(parameters)[0]
        f= np.zeros(len(x))
        for i in range(len(x)):
            e = eval_ind(x[i])
            f[i]=e  # Evalúa el string con las variables de x
        len(f)
        out["F"] =  f

# %%
problem = MyProblem()

# Configurar el algoritmo PSO
algorithm = PSO(
    pop_size=50,  # Tamaño de la población
    w=0.9,  # Factor de inercia
    c1=0.5,  # Componente cognitiva
    c2=0.3,  # Componente social
)


# Ejecutar la optimización
result = minimize(
    problem,
    algorithm,
    termination=("n_gen", 100),  # Número de generaciones
    seed=42,  # Para reproducibilidad
    verbose=True,
)

# Show results
print("Best solution found:", result.X)
print("Objective function value:", result.F)

# %%
result.X

# %%
# sim_time= 2500
# eq_energy_level = equall_dep_pot_energy_level(eta=eta,
#                                               alpha=alpha,
#                                               a_h=100)
# sr_pre, sr_post, mult, wr, fin_weights = experiment(epochs=1,
#                                        poisson_rate=10,
#                                        n_pre_neurons=n_pre_neurons,
#                                        sim_time=sim_time,
#                                        syn_spec=syn_spec)


# plots(mult=mult, 
#       wr=wr, 
#       eta=eta, 
#       fin_weights=fin_weights, 
#       eq_energy_level=eq_energy_level)


