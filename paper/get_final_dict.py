
from mpi4py import MPI
import numpy
import os, sys, pickle

def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N))
    into equal length chunks. Order is not preserved.
    """
    return [container[_i::count] for _i in range(count)]

models = range(100)

COMM = MPI.COMM_WORLD

if COMM.rank == 0:
    splitted_jobs = split(models, COMM.size)
    print("Split selected_variables = {}".format(splitted_jobs))
else:
    splitted_jobs = None

scattered_jobs = COMM.scatter(splitted_jobs, root=0)

results = []
for n_model in scattered_jobs:
    results_file_dict = "./experiment/dm_methods/results/final_results_synthetic_{}.pkl".format(n_model)

    with open(results_file_dict, "rb") as f:
        eval = pickle.load(f)

    results.append(eval.metrics)

results = MPI.COMM_WORLD.gather(results, root=0)

# TO POLISH
if COMM.rank == 0:
    # Collect all results in dictionaries
    #
    print("\nCollecting results...")

    methods = ('pcmci', 'corr', 'varimax_pcmci_w', 'varimax_corr_w', 'pca_pcmci_w', 'pca_corr_w')
    metrics = ('grid_mse', 'grid_rmae', 'grid_precision', 'grid_recall')

    dict_results = {meth : {metr: [] for metr in metrics} for meth in methods}

    for file in results:
        for res in file:
            for meth in methods:
                for metr in metrics:
                    dict_results[meth][metr].append(res[meth][metr])

    with open("joined_results_synthetic.pkl", "wb") as f:
        pickle.dump(dict_results, f, protocol=5)

