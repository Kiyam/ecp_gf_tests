import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
import seaborn as sns
from pathlib import Path

def plot_goodness_of_fit(log_vals, data, dof, save_path = None):
    cm = mpcm.get_cmap('plasma')
    colors = [cm(x) for x in np.linspace(0.01, 0.75, 5)]

    log_vals = log_vals*-2/dof
    data = data*-2/dof

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(log_vals)
    plt.axvline(data, 0, 1.6, linestyle='--', color=colors[2])
    plt.xlabel(r'$\chi^2_\mathrm{red}$', fontsize=16)
    plt.ylabel(r'$P(\chi^2_\mathrm{red}) $', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)