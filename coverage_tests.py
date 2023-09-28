import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from pathlib import Path

def calc_dists(samples, truth):
    """
    Takes an array of samples and the "truth", calculates the Euclidean distance from the truth to the samples to return a f_metric count
    """
    random_index = np.random.randint(0, len(samples))
    ref_point = samples[random_index]
    samples = np.delete(samples, random_index, axis=0)
    ref_dist = distance.euclidean(ref_point,truth)
    distances = distance.cdist(samples, np.expand_dims(ref_point,0)).flatten()
    f_metric = (distances>ref_dist).sum()/len(distances)
    return f_metric

def coverage(sim_samples, truth, alpha_levels = 101):
    """
    Takes an array of many samples with many truths, and computes the expected coverage probability for many alpha levels across all of those samples
    """
    assert truth.shape[0] == sim_samples.shape[0], "Number of truths should equal the number of sets of samples from simulations on those truths"
    alphas = np.linspace(0,1,alpha_levels)
    ecp = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        f_metrics = np.zeros(num_sims)
        actual_num_sims = 0
        for sim_num in range(len(sim_samples)):
            f_metrics[sim_num] += calc_dists(sim_samples, truth[sim_num])
            actual_num_sims += 1
        ecp[i] += (f_metrics<1-alpha).sum()/actual_num_sims
    return alphas, ecp

def plot_coverage(alphas, ecp, save_path = None):
    fig, ax = plt.subplots(figsize=(12, 8))
    cm = mpcm.get_cmap('plasma')
    colors = [cm(x) for x in np.linspace(0.01, 0.75, 5)]
    x = 1-alphas
    y = ecp
    ax.scatter(x, y, marker = '.', color = colors[4])
    ax.plot(x, x, '--', color = colors[0])
    plt.xlabel("Credibility Level", fontsize=16)
    plt.ylabel("Expected Coverage", fontsize=16)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=14)
    plt.title('Glass SBI')
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

if __name__ == "__main__":
    sim_samples = np.load('path_to_sets_of_sim_samples') # Needs to be of dimensionality N_Truths by N_params by N_samples
    truth = np.load('path_to_true_param_for_each_sample_set')
    alphas, ecp = coverage(sim_samples = sim_samples, truth = truth)
    plot_coverage(alphas = alphas, ecp = ecp, save_path = "path_to_graph")