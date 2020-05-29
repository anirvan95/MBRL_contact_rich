import pickle
from common.clustering_utils import hybridSegmentClustering
import matplotlib.pyplot as plt

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

clustering_params = {
    'per_train': 1,  # percentage of total rollouts to be trained
    'window_size': 2,  # window size of transition point clustering
    'weight_prior': 0.05,  # weight prior of DPGMM clustering for transition point
    'DBeps': 3.0,  # DBSCAN noise parameter for clustering segments
    'DBmin_samples': 2  # DBSCAN minimum cluster size parameter for clustering segments
}

p = pickle.load(open("option_critic_data.pkl", "rb"))
rows = 8
cols = 10
for epoch in range(9, len(p)):
    print('Iteration number', epoch)
    plt_name = "/home/anirvan/MasterThesis/MBRL_contact_rich/logs/optionClusterPlt_"+str(epoch)+".png"
    data = p[epoch]
    rollouts = data['rollouts']
    nmodes, segmentedRollouts, x_train, u_train, delx_train, label_train, label_t_train = hybridSegmentClustering(rollouts, clustering_params)
    for rollout in range(0, len(segmentedRollouts)):
        plt.subplot(rows, cols, rollout + 1)
        x_data = segmentedRollouts[rollout]['x']
        label_data = segmentedRollouts[rollout]['label']
        for t in range(0, len(x_data)):
            if label_data[t] == 0:
                plt.plot(t, x_data[t, 0], 'r.')
            elif label_data[t] == 1:
                plt.plot(t, x_data[t, 0], 'b.')
            else:
                plt.plot(t, x_data[t, 0], 'k.')
    #plt.show()
    plt.savefig(plt_name)
    plt.clf()

