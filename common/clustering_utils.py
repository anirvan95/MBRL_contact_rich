import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import sys
from math import sqrt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def computeDistance(f1, f2):
    dim = int(0.5 * (-1 + sqrt(1 + 4 * len(f1))))
    mf1 = f1[0:dim]
    mf2 = f2[0:dim]
    covf1 = np.reshape(f1[dim:], (-1, dim))
    covf2 = np.reshape(f2[dim:], (-1, dim))
    return .5 * (bhattacharyyaGaussian(mf1, covf1, mf2, covf2) + bhattacharyyaGaussian(mf2, covf2, mf1, covf1))


def bhattacharyyaGaussian(pm, pv, qm, qv):
    """
    Computes Bhattacharyya distance between two Gaussians
    with diagonal covariance.
    """
    # Difference between means pm, qm
    diff = np.expand_dims((qm - pm), axis=1)
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = np.linalg.det(pv)
    ldqv = np.linalg.det(qv)
    # Log-determinant of pqv
    ldpqv = np.linalg.det(pqv)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * np.log(ldpqv / (np.sqrt(ldpv * ldqv)))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    temp = np.matmul(diff.transpose(), np.linalg.pinv(pqv))
    dist = 0.125 * np.matmul(temp, diff)
    return np.float(dist + norm)


def smoothing(indices):
    """
        Smoothing for transition point detection [IMPROVE]
    """
    newIndices = indices
    for i in range(1, len(indices) - 1):
        if indices[i] != indices[i - 1] and indices[i] != indices[i + 1] and indices[i + 1] == indices[i - 1]:
            newIndices[i] = indices[i + 1]

    return newIndices


def identifyTransitions(traj, window_size, weight_prior):
    """
        Identify transition by accumulating data points using sliding window and using DP GMM to find
        clusters in a single trajectory
    """
    total_size = traj.shape[0]
    dim = traj.shape[1]
    demo_data_array = np.zeros((total_size - window_size, dim * window_size))
    inc = 0
    for i in range(window_size, total_size):
        window = traj[i - window_size:i, :]
        demo_data_array[inc, :] = np.reshape(window, (1, dim * window_size))
        inc = inc + 1

    estimator = BayesianGaussianMixture(n_components=2, n_init=10, max_iter=300,
                                        weight_concentration_prior=weight_prior,
                                        init_params='random', verbose=False)
    labels = estimator.fit_predict(demo_data_array)
    # print(estimator.weights_)
    filtabels = smoothing(labels)
    # print(labels)
    inc = 0
    transitions = []
    for j in range(window_size, total_size):

        if inc == 0 or j == window_size:
            pass  # self._transitions.append((i,0))
        elif j == (total_size - 1):
            pass  # self._transitions.append((i,n-1))
        elif filtabels[inc - 1] != filtabels[inc]:
            transitions.append(j - window_size)
        inc = inc + 1

    transitions.append(0)
    transitions.append(total_size - 1)
    transitions.sort()

    # print("[TSC] Discovered Transitions (number): ", len(transitions))
    return transitions


def fitGaussianDistribution(traj, action, transitions):
    """
        Fits gaussian distribution in each segment of the trajectory
    """
    nseg = len(transitions)
    dim = traj.shape[1]
    dynamicMat = []
    rmse = 0
    selectedSeg = []
    for k in range(0, nseg - 1):
        if transitions[k + 1] - transitions[k] > 8:
            # ensuring at least one sample is there between two transition point
            x_t_1 = traj[(transitions[k] + 1):(transitions[k + 1] + 1), :]
            x_t = traj[transitions[k]:transitions[k + 1], :]
            u_t = action[transitions[k]:transitions[k + 1], :]
            feature_data_array = np.hstack((x_t, x_t_1))
            meanGaussian = np.mean(feature_data_array, axis=0)
            covGaussian = np.cov(feature_data_array, rowvar=0)
            covGaussian = covGaussian + covGaussian.T
            covFeature = covGaussian.flatten()
            det = np.linalg.det(covGaussian)
            if np.linalg.cond(covGaussian) < 1 / sys.float_info.epsilon:
                # print("Segment Number: ", k)
                selectedSeg.append(np.array([transitions[k], transitions[k + 1]]))
                dynamicMat.append(np.append(meanGaussian, covGaussian))
            else:
                print("Singular Matrix !!! ")

    return np.array(dynamicMat), np.array(selectedSeg)


def correctMode(segTraj, mode):
    rectBox = [-0.125, 0.725, 0.4, 0.75]
    points_in = 0
    correctedMode = 0 # default mode
    if mode > 0:
        for t in range(0, len(segTraj)):
            points_in = points_in + (rectBox[0] < segTraj[t, 0] < rectBox[0]+rectBox[2] and rectBox[1]-rectBox[3] < segTraj[t, 1] < rectBox[1])
        if points_in > 0.7*len(segTraj):
            correctedMode = mode

    return correctedMode


def hybridSegmentClustering(rollouts, clustering_params):
    total_rollouts = len(rollouts)
    train_rollouts = int(clustering_params['per_train']*total_rollouts)
    trajMat = []
    segmentedData = []
    for rollout in range(0, train_rollouts):
        traj = rollouts[rollout]
        states = traj['observations']
        action = traj['actions']
        state_scaler = StandardScaler().fit(states)
        state_std = state_scaler.transform(states)
        time_vect = np.expand_dims(np.arange(0, len(states)), axis=1)
        time_scaler = StandardScaler().fit(time_vect)
        time_std = time_scaler.transform(time_vect)
        feature_vect = np.hstack((time_std, state_std))
        tp = identifyTransitions(state_std, clustering_params['window_size'], clustering_params['weight_prior'])
        # fitting Gaussian Dynamic model
        fittedModel, selTraj = fitGaussianDistribution(states[:, 0:2], action, tp)
        trajMat.append(np.array([rollout, selTraj]))
        if rollout == 0:
            dynamicMat = fittedModel
        else:
            dynamicMat = np.concatenate((dynamicMat, fittedModel), axis=0)
        rollout += 1

    trajMat = np.array(trajMat)
    db = DBSCAN(eps=clustering_params['DBeps'], min_samples=clustering_params['DBmin_samples'], metric=computeDistance)
    labels = db.fit_predict(dynamicMat)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    segCount = 0
    for rollout in range(0, train_rollouts):
        traj = rollouts[rollout]
        states = traj['observations']
        action = traj['actions']
        delta_traj = []
        for t in range(len(states) - 1):
            delta_states = states[t + 1, :] - states[t, :]
            delta_traj.append(delta_states)
        delta_traj.append(delta_states)
        delta_traj = np.array(delta_traj)
        for segment in range(0, trajMat[rollout][1].shape[0]):
            segTraj = states[trajMat[rollout][1][segment][0]:(trajMat[rollout][1][segment][1] + 1), :]
            segAction = action[trajMat[rollout][1][segment][0]:(trajMat[rollout][1][segment][1] + 1), :]
            segDelta = delta_traj[trajMat[rollout][1][segment][0]:(trajMat[rollout][1][segment][1] + 1), :]
            labels[segCount] = correctMode(segTraj, labels[segCount])

            if labels[segCount] >= 0: #ignoring noisy clusters

                if segment == 0:
                    segLabel = labels[segCount] * np.ones((len(segTraj), 1))
                    x_data = segTraj
                    u_data = segAction
                    delta_data = segDelta
                    label_data = labels[segCount] * np.ones((len(segTraj), 1))
                else:
                    segLabel = np.vstack((segLabel, labels[segCount] * np.ones((len(segTraj), 1))))
                    x_data = np.vstack((x_data, segTraj))
                    u_data = np.vstack((u_data, segAction))
                    delta_data = np.vstack((delta_data, segDelta))
                    label_data = np.vstack((label_data, labels[segCount] * np.ones((len(segTraj), 1))))

            segCount = segCount + 1

        segLabel_t_data = segLabel
        for i in range(len(segLabel) - 1):
            segLabel_t_data[i] = segLabel[i + 1]

        label_t_data = segLabel_t_data
        dict = {'x': x_data, 'u': u_data, 'delta': delta_data, 'label': label_data, 'label_t': label_t_data}
        segmentedData.append(dict)

        firstDataFlag = True
        for rollout in range(0, len(segmentedData)):
            if firstDataFlag:
                x_train = segmentedData[rollout]['x']
                u_train = segmentedData[rollout]['u']
                delx_train = segmentedData[rollout]['delta']
                label_train = segmentedData[rollout]['label']
                label_t_train = segmentedData[rollout]['label_t']
                firstDataFlag = False
            else:
                x_train = np.vstack((x_train, segmentedData[rollout]['x']))
                u_train = np.vstack((u_train, segmentedData[rollout]['u']))
                delx_train = np.vstack((delx_train, segmentedData[rollout]['delta']))
                label_train = np.vstack((label_train, segmentedData[rollout]['label']))
                label_t_train = np.vstack((label_t_train, segmentedData[rollout]['label_t']))

    n_modes = len(np.unique(labels))
    print('Corrected number of modes: %d' % n_modes)
    return n_modes, segmentedData, x_train, u_train, delx_train, label_train, label_t_train