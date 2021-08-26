import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np
import pickle

if __name__ == "__main__":
    files = {"synthetic": "joined_results_synthetic.pkl",
             "low_resolution": "joined_results_small.pkl",
             "large_resolution": "joined_results.pkl"}

    for k, f in files.items():
        with open(f, "rb") as f:
            files[k] = pickle.load(f)

    methods = ('corr', 'pcmci', 'pca_corr_w', 'pca_pcmci_w', 'varimax_corr_w',   'varimax_pcmci_w')
    metrics = ('grid_mse', 'grid_rmae', 'grid_precision', 'grid_recall')
    metrics_plot = ('grid_mse', 'grid_precision', 'grid_recall')

    colors = ['cornflowerblue', 'sandybrown', 'indianred', 'purple', 'brown', 'green']

    dict_results = {k: {metr: {meth: [] for meth in methods} for metr in metrics} for k in files.keys()}
    dict_labels = {
        "corr": "$\mathbf{C}$",
        "pcmci": "$\mathbf{P}$",
        'varimax_pcmci_w': "$\mathbf{V^+P}$",
        "varimax_corr_w": "$\mathbf{V^+C}$",
        'pca_pcmci_w': "$\mathbf{P^{ca}P}$",
        'pca_corr_w': "$\mathbf{P^{ca}C}$",
    }

    dict_titles = {
        0: r"$\bf{Synthetic}$ $\bf{data}$",
        1: r"$\bf{Surface}$ $\bf{Pressure}$ $\bf{dataset}$ $\bf{(low}$ $\bf{resolution)}$",
        2: r"$\bf{Surface}$ $\bf{Pressure}$"
    }

    row_dict = {
        0: "synthetic",
        1: "low_resolution",
        2: "large_resolution"
    }

    y_labels = ("MSE$^\mathcal{E}$", "$Pr$^\mathcal{M}$", "$Re$^\mathcal{M}$")

    for k, res in dict_results.items():
        for metr in metrics:
            for meth in methods:
                dict_results[k][metr][meth] = files[k][meth][metr]

    fig = plt.figure(constrained_layout=True)
    fig.tight_layout()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    # fig.suptitle('Grid level results')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(dict_titles[row])

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)
        bp = []

        for col, ax in enumerate(axs):
            metr = metrics_plot[col]
            labels = dict_results[row_dict[row]][metr].keys()
            labels = [dict_labels[l] for l in labels]
            data = dict_results[row_dict[row]][metrics_plot[col]].values()
            data = [[m for m in d if str(m) != 'nan'] for d in data]

            bp.append(ax.boxplot(data,
                                 vert=True,
                                 patch_artist=True,
                                 labels=labels)
                      )

            ax.set_ylabel(y_labels[col])

            if metr in ["grid_mse"]:
                ax.set_yscale('log')
            #ax.set_title(f'Plot title {col}')

        for box_plot in bp:
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)

    plt.savefig('grid_results.png')
    plt.show()

    # OLD
    test = False
    if test:
        metrics_plot = ('grid_mse', 'grid_precision', 'grid_recall')

        fig, ax = plt.subplots(nrows=1, ncols=3)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        bp = []

        for i, a in enumerate(ax):
            metr = metrics_plot[i]
            labels = dict_results[metr].keys()
            labels = [dict_labels[l] for l in labels]
            data = dict_results[metr].values()
            data = [[m for m in d if str(m) != 'nan'] for d in data]

            bp.append(a.boxplot(data,
                                vert=True,
                                patch_artist=True,
                                labels=labels)
                      )
            fig.tight_layout()
            if metr in ["grid_mse"]:
                a.set_yscale('log')

        for box_plot in bp:
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)

        plt.show()


