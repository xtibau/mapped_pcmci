# This file creates the datasets for the experiments presented in the paper
import pickle
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mapped_pcmci.savar_extractor import SavarExtractor
from varimax_plus.varimax_plus import Varimax, VarimaxPlus

if __name__ == "__main__":

    print(os.getcwd())

    # Config
    generate_models = False
    save_params = True
    load_params = False
    n_models = 100
    folder_save_models = "./experiment/data/"
    same_cg = True  # If we will perturb Phi for each model.

    if not load_params:

        savar_extractor = SavarExtractor(verbose="DEBUG", threshold_graph=0.1, var_alpha_level=0.001,
                                         max_comps=10)
        savar_extractor.from_nc("./data/pres.sfc.anommonth.1948-2014r53_27.nc")
        savar_extractor.get_savar_dict()

        # Get SAVAR
        savar_model = savar_extractor.generate_savar_model()
        savar_model.noise_cov = None
        savar_model.fast_noise_cov = np.eye(savar_model.spatial_resolution)
        savar_model.noise_strength = 20

        for i in range(10):
            plt.imshow(savar_model.mode_weights[i, ...])
            plt.show()

        # Save savar model
        if save_params:
            with open("experiment/data/savar_model_small.pkl", "wb") as f:
                pickle.dump(savar_model, f, protocol=5)

    else:
        with open("experiment/data/savar_model_small.pkl", "rb") as f:
            savar_model = pickle.load(f)

    if generate_models:
        for n in range(n_models):
            if same_cg:
                savar_model.time_length = 500
                new_model = deepcopy(savar_model)
                new_model.generate_data()
                model_name = "savar_model_{}.pkl".format(n)
                with open(folder_save_models + model_name, "wb") as f:
                    pickle.dump(new_model, f, protocol=5)

    test = False
    if test:
        for i in range(10):
            plt.imshow(savar_model.mode_weights[i, ...])
            plt.show()


        var = Varimax(savar_model.data_field.T, max_comps=10)
        var_results = var()

        for i in range(10):
            plt.imshow(var_results["weights"][..., i].reshape(27, 53))
            plt.show()

        var_plus = VarimaxPlus(savar_model.data_field.T, max_comps=10,
                               boot_rep=500, alpha_level=0.01, boot_samples=1000)
        var_plus_results = var_plus()

        # Test different alpha components
        var_plus.alpha_level = 0.001
        var_plus_results = var_plus.results

        # Mask to see the Os
        masked_array = np.ma.masked_where(var_plus_results["weights"] == 0, var_plus_results["weights"])

        for i in range(10):
            plt.imshow(masked_array[..., i].reshape(27, 53))
            plt.show()