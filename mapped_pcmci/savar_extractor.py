# import nctoolkit as nct
# nct.deep_clean()
import netCDF4 as nc
import numpy as np
from copy import deepcopy
from varimax_plus.varimax_plus import VarimaxPlus
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from tigramite.models import LinearMediation
import tigramite.data_processing as pp

from savar.savar import SAVAR

from mapped_pcmci.functions import threshold_link_strenght

class SavarExtractor:
    """
    Extracts savar parameters from a given ncl file
    """

    savar_config = {
        "links_coeffs": None,
        "time_length": 500,
        "mode_weights": None,
        "transient": 200,
        "noise_weights": None,
        "noise_strength": 1,
        "noise_variance": 1,
        "noise_cov": None,
        "latent_noise_cov": None,
        "fast_cov": None,
        "forcing_dict": None,
        "season_dict": None,
        "data_field": None,
        "noise_data_field": None,
        "seasonal_data_field": None,
        "forcing_data_field": None,
        "linearity": "linear",
        "verbose": False,
        "model_seed": None,
    }

    def __init__(self, var: str = "pres", cos_weight: bool = True, truncate_by='max_comps', max_comps=60,
                 fraction_explained_variance=0.9, boot_axis: int = 0, boot_rep: int = 100,
                 boot_samples: int = 100, var_alpha_level: float = 0.01, threshold_graph: float = 0.15,
                 pcmci_significance: str = 'analytic', pcmci_ind_test: str = "ParCorr", pcmci_tau_max: int = 3,
                 pcmci_pc_alpha: float = 0.2, pcmci_parents_alpha: float = 0.05,
                 verbose: str = None):

        # nc_file
        self.var = var

        # Varimax
        self.cos_weight = cos_weight
        self.truncate_by = truncate_by
        self.max_comps = max_comps
        self.fraction_explained_variance = fraction_explained_variance
        self.boot_axis = boot_axis
        self.boot_rep = boot_rep
        self.boot_samples = boot_samples
        self.var_alpha_level = var_alpha_level
        self.threshold_graph = threshold_graph
        self.pcmci_parents_alpha = pcmci_parents_alpha

        # PCMCI
        self.pcmci_significance = pcmci_significance
        self.pcmci_ind_test = pcmci_ind_test
        self.pcmci_tau_max = pcmci_tau_max
        self.pcmci_pc_alpha = pcmci_pc_alpha

        self.verbose = verbose

        # Empty attributes
        self.lat_weights = None
        self.var_results = None
        self.dimensions = None
        self.weights = None
        self.causal_graph = None
        self.noise_cov = None

        # For testing
        self.varimax_computed = False
        self.cov_computed = False

    def from_nc(self, file: str):
        """

        :param file: nc file
        :return:
        """

        self.nc_dataset = nc.Dataset(file)

        if self.verbose in ["INFO", "DEBUG"]:
            print(self.nc_dataset)

    def get_savar_dict(self, ):

        # Get Varimax weights
        self._get_varimax_weights()

        if self.verbose == "DEBUG":
            print("varimax+ performed")

        # Weights now lat*lon x comps
        self.var_results["weights"] = self.var_results["weights"].transpose()  # need to be transposed

        # Weights now comps x lat x lon
        _, lat, lon = self.dimensions
        self.weights = self.var_results["weights"].reshape(-1, lat, lon)

        self.savar_config['mode_weights'] = deepcopy(self.weights)

        # Get components
        self.components = self.dataset.reshape(-1, lat * lon) @ self.var_results["weights"].transpose()

        # Get links of the components
        pcmci_config = {
            "significance": self.pcmci_significance,
            "ind_test": self.pcmci_ind_test,
            "tau_max": self.pcmci_tau_max,
            "pc_alpha": self.pcmci_pc_alpha,
            "parents_alpha": self.pcmci_parents_alpha,
            "verbose": True if self.verbose == "DEBUG" else False,
        }
        # Data must be T x L
        if self.verbose in ["INFO", "DEBUG"]:
            print("Performing pcmci")
        self.pcmci_results = self._get_pcmci_results(self.components, **pcmci_config)

        # Estimated causal graph in Tigramite notation
        self.causal_graph = self.from_phi_to_cg(self.pcmci_results["phi_estimated"])
        if self.causal_graph is not None:
            self.causal_graph = threshold_link_strenght(self.causal_graph, self.threshold_graph)
        self.savar_config['links_coeffs'] = self.causal_graph

        Y = self.dataset.reshape(-1, lat * lon).transpose()
        W = self.weights.reshape(-1, lat * lon)
        Phi = self.pcmci_results["phi_estimated"]
        if self.threshold_graph is not None:
            Phi[np.abs(Phi) <= self.threshold_graph] = 0
        if not self.cov_computed:  # Don't do it again.
            self.noise_cov = self.get_covariance_e_y(Y, W, Phi)
        self.savar_config["noise_cov"] = self.noise_cov

    def _get_varimax_weights(self):

        var_dict = {
            "truncate_by": self.truncate_by,
            "max_comps": self.max_comps,
            "fraction_explained_variance": self.fraction_explained_variance,
            "boot_axis": self.boot_axis,
            "boot_rep": self.boot_rep,
            "boot_samples": self.boot_samples,
            "alpha_level": self.var_alpha_level,
            "verbose": True if self.verbose == "DEBUG" else False
        }

        if self.cos_weight:
            self.lat_weights = np.cos(np.deg2rad(self.nc_dataset.variables["lat"][:]))
            dataset = deepcopy(self.nc_dataset.variables[self.var][:])
            dataset = np.einsum('i,tij->tij', self.lat_weights, dataset)  # Multiply each value along lat weights
            self.dataset = dataset

        else:
            self.dataset = deepcopy(self.nc_dataset.variables[self.var][:])

        dataset = deepcopy(self.dataset)  # time x Lat x Lon
        self.dimensions = dataset.shape
        n_time, lat, lon = self.dimensions
        dataset = dataset.reshape(n_time, lat * lon)

        if not self.varimax_computed:
            if self.verbose in ["INFO", "DEBUG"]:
                print("Doing varimax!")
            var_p = VarimaxPlus(data=dataset, **var_dict)
            var_p.varimax_plus()
            self.varimax_computed = True
            self.var_results = var_p.results

    def get_covariance_e_y(self, Y, W, Phi):
        """
        From a SAVAR model Y = W^+\sum\PhiWx +e_y, returns e_y by extracting
        :param Y: The dataset at grid level. Shape
        :param X: W @ The dataset at grid level
        :param W: The estimated weights
        :return: Y - W@X
        """
        _, lat, lon = self.dimensions
        tau_max = Phi.shape[0]  # Phi must be tau_max, N, N
        # Its original (t, lat, lon) we need it (lon*lat, t)
        Y = deepcopy(Y).reshape(-1, lat * lon).transpose()
        time_length = Y.shape[1]
        W_plus = np.linalg.pinv(W)
        Y_pred = np.zeros_like(Y)
        if self.verbose in ["INFO", "DEBUG"]:
            print("Computing covariance matrix of the results.")
        for t in range(tau_max, time_length):
            for i in range(tau_max):
                Y_pred[..., t:t + 1] = W_plus @ Phi[1, ...] @ W @ Y[..., t - 1 - i:t - i]

        return np.cov((Y - Y_pred)[..., tau_max:])

    def generate_savar_model(self, **kwargs) -> SAVAR:
        """
        Creates a SAVAR model from the extracted dataset
        :param kwargs: Used to pass SAVAR parameters to overwrite the default parameters
        :return: savar.savar.SAVAR
        """

        savar_config = deepcopy(self.savar_config)

        # For each parameter of SAVAR we check if there is another value in kwargs, if not, we use the default
        default_values = [(key, val) for key, val in savar_config.items()]

        # Change the savar dict only if there is a new parameter provided to this function
        for (key, default) in default_values:
            self.check_value(savar_config, kwargs, key, default)

        return SAVAR(**savar_config)

    @staticmethod
    def check_value(odict, kwargs, key, default):
        odict[key] = kwargs.get(key, default)

    @staticmethod
    def _get_pcmci_results(data: np.ndarray, significance: str = 'analytic', ind_test: str = "ParCorr",
                           tau_max: int = 3, pc_alpha: float = 0.2, parents_alpha: float = 0.05,
                           verbose: bool = True) -> dict:
        """
        Get an estimation of \Phi using the Pcmci algorithm
        :param data: Input in the shape T x L
        :param significance:
        :param ind_test:
        :param tau_max:
        :param pc_alpha:
        :param parents_alpha:
        :param verbose:
        :return:
        """

        if significance == 'analytic' and ind_test == "ParCorr":
            ind_test = ParCorr(significance=significance)
        else:
            raise NotImplementedError("Only ParCorr test is implemented. You select:"
                                      " {} is not yet implemented".format(ind_test))

        # Observations. We don't do the others because we don't estimate them again.
        pcmci = deepcopy(PCMCI(
            # Input data for PCMCI T times L
            dataframe=pp.DataFrame(data),
            cond_ind_test=ind_test,
            verbosity=verbose,
        ))

        pcmci_results = pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)

        # Get the parents
        parents_predicted = pcmci.return_significant_links(
            pq_matrix=pcmci_results["p_matrix"],
            val_matrix=pcmci_results["val_matrix"],
            alpha_level=parents_alpha,
            include_lagzero_links=False,
        )["link_dict"]

        dataframe = deepcopy(pcmci.dataframe)
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=parents_predicted, tau_max=tau_max)
        phi_estimated = med.phi[1:, ...]  # Remove the self links at tau=0

        return {
            "pcmci": pcmci,
            "pcmci_results": pcmci_results,
            "parents": parents_predicted,
            "phi_estimated": phi_estimated  # tau, i, j
        }

    @staticmethod
    def from_phi_to_cg(phi: np.ndarray) -> dict:
        """
        from a guven phi returns the causal grap
        :param phi: Starting at tau = 1, of shape tau, i, j
        :return:
        """

        if phi.ndim != 3:
            raise ValueError("Phi must be a three dimensional array. Now is: {}".format(phi.ndim))

        m_tau, m_i, m_j = phi.shape
        causal_graph = {}

        for i in range(m_i):
            causal_graph[i] = []
            for j in range(m_j):
                for tau in range(m_tau):
                    if phi[tau, i, j] == 0:
                        continue
                    else:
                        causal_graph[i].append(((j, -(tau + 1)), phi[tau, i, j]))

        return causal_graph
