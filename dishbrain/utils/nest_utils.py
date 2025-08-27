import nest 

def setup_nest(sim_dict, module_name="edlif_psc_alpha_0_module"):
    try:
        nest.Install(module_name)
    except:
        pass
    # Setup nest
    nest.ResetKernel()
    nest.local_num_threads = sim_dict['local_num_threads']
    nest.resolution = sim_dict['sim_resolution']
    nest.rng_seed = sim_dict['rng_seed']
    nest.overwrite_files = sim_dict['overwrite_files']
    nest.print_time = sim_dict['print_time']
    
    if nest.Rank() == 0:
        print('RNG seed: {}'.format(
            nest.rng_seed))
        print('Total number of virtual processes: {}'.format(
            nest.total_num_virtual_procs))