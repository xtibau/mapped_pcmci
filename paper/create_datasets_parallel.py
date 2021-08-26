# This file generates datasets from the stored savar model

import pickle
from copy import deepcopy
from mpi4py import MPI

from savar.model_generator import SavarGenerator

# Default communicator
COMM = MPI.COMM_WORLD

# Some functions
def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N))
    into equal length chunks. Order is not preserved.
    """
    return [container[_i::count] for _i in range(count)]

use_sinthetyc = True

if use_sinthetyc:
    savar_gen = SavarGenerator(resolution=(15, 45))
    savar_model = savar_gen.generate_savar()
    folder_save_models = "./experiment/data_synthetic/"
n_models = 100


if not use_sinthetyc:
    savar_file = "./experiment/data/savar_model_small.pkl"
    folder_save_models = "./experiment/data_small/"

    with open(savar_file, "rb") as f:
        savar_model = pickle.load(f)


# The master
if COMM.rank == 0:
    splitted_jobs = split(range(n_models), COMM.size)
else:
    splitted_jobs = None

scattered_jobs = COMM.scatter(splitted_jobs, root=0)

for n in scattered_jobs:

    print("Starting job {}".format(n))

    savar_model.time_length = 1000
    new_model = deepcopy(savar_model)
    new_model.generate_data()
    model_name = "savar_model_synthetic_{}.pkl".format(n)

    with open(folder_save_models + model_name, "wb") as f:
        pickle.dump(new_model, f, protocol=5)
