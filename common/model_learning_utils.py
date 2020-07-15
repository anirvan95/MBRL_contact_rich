import numpy as np
import gpflow
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import BayesianGaussianMixture
from math import sqrt
import sys


def obtainMode(env_id, point):
    if env_id == 'Block2D-v2':
        Mode_11 = [-0.1, 0.55, -0.05, 0.75]
        Mode_12 = [-0.15, 0.25, 0.3, 0.75]
        Mode_13 = [-0.05, 0.75, 1.15, 0.8]
        Mode_21 = [-0.08, 0.45, 0.05, 0.55]
        mode = 0
        if Mode_11[0] <= point[0] <= Mode_11[2] and Mode_11[1] <= point[1] <= Mode_11[3]:
            mode = 1
        if Mode_12[0] <= point[0] <= Mode_12[2] and Mode_12[1] <= point[1] <= Mode_12[3]:
            mode = 1
        if Mode_13[0] <= point[0] <= Mode_13[2] and Mode_13[1] <= point[1] <= Mode_13[3]:
            mode = 1
        if Mode_21[0] <= point[0] <= Mode_21[2] and Mode_21[1] <= point[1] <= Mode_21[3]:
            mode = 1

        return mode

    elif env_id == 'Block3D-v1':
        box_size = 0.15
        bound = box_size / 2 + 0.001
        Mode_1 = [0, 1, 0, 1, 0, bound]
        Mode_2 = [0, 1, 0, bound, bound, 1]
        Mode_3 = [0, bound, bound, 1, bound, 1]
        mode = 0
        if Mode_1[0] <= point[0] <= Mode_1[1] and Mode_1[2] <= point[1] <= Mode_1[3] and Mode_1[4] <= point[2] <= \
                Mode_1[5]:
            mode = 1
        elif Mode_2[0] <= point[0] <= Mode_2[1] and Mode_2[2] <= point[1] <= Mode_2[3] and Mode_2[4] <= point[2] <= \
                Mode_2[5]:
            mode = 2
        elif Mode_3[0] <= point[0] <= Mode_3[1] and Mode_3[2] <= point[1] <= Mode_3[3] and Mode_3[4] <= point[2] <= \
                Mode_3[5]:
            mode = 3

        return mode


class multiDimGaussianProcess(object):
    '''
    Trains Multi Dimensional GP's for return maps and mode dynamics
    :param gpr_params: gpflow gpr params
    '''

    def __init__(self, gpr_params):
        self.gp_param = gpr_params

    def fit(self, X, Y):
        assert(Y.shape[1] >= 2)
        assert (X.shape[1] >= 2)
        self.gp_list = []
        in_dim = X.shape[1]
        out_dim = Y.shape[1]
        self.out_dim = out_dim

        #TODO: Possible parallelization here
        for i in range(self.out_dim):
            gp_params = self.gp_param
            normalize = gp_params['normalize']
            y = Y[:, i].reshape(-1, 1)
            x_sig = np.sqrt(np.var(X, axis=0))
            len_scale = x_sig
            # print('init_len_scale', len_scale)
            len_scale_lb = np.min(x_sig * gp_params['ls_b_mul'][0])
            len_scale_ub = np.max(x_sig * gp_params['ls_b_mul'][1])
            len_scale_b = (len_scale_lb, len_scale_ub)
            y_var = np.var(Y[:, i])
            sig_var = y_var
            noise_var = 1e-5
            k = gpflow.kernels.SquaredExponential(lengthscales=tf.Variable((len_scale)),
                                                  variance=tf.Variable((sig_var)))

            m = gpflow.models.GPR(data=(X, y), kernel=k, noise_variance=noise_var)
            gpflow.utilities.set_trainable(m.likelihood.variance, False)

            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

            self.gp_list.append(m)

    def predict(self, X, return_std=True):
        Y_mu = np.zeros((X.shape[0], self.out_dim))
        Y_std = np.zeros((X.shape[0], self.out_dim))
        for i in range(self.out_dim):
            gp = self.gp_list[i]
            mu, var = gp.predict_f(X)
            Y_mu[:, i] = np.squeeze(mu)
            Y_std[:, i] = np.sqrt(var).reshape(-1)
        return Y_mu, Y_std


class SVMPrediction(object):
    '''
    Trains SVMs for interest and guard functions
    :param svm_grid_params: parameters for grid search
    :param svm_params: parameters for training
    '''
    def __init__(self, svm_params, svm_grid_params):
        self.svm_params = svm_params
        self.svm_grid_params = svm_grid_params
        self.clf = SVC(**self.svm_params)
        self.grid_search = GridSearchCV(self.clf, **self.svm_grid_params)

    def train(self, X, y, grid_search=False):
        if grid_search:
            print("Performing grid search")
            self.grid_search.fit(X, y)
            self.svm_params.update(self.grid_search.best_params_)
            print("Params:", self.svm_params)
            self.clf = SVC(**self.svm_params)
            self.clf.fit(X, y)
        else:
            self.clf = SVC(**self.svm_params)
            self.clf.fit(X, y)

    def predict(self, X):
        mode = self.clf.predict(X)
        return mode

    def predict_f(self, X):
        mode_prob = self.clf.predict_proba(X)
        return mode_prob


class LRPrediction(object):
    '''
    Trains Logistic Regression for interest and guard functions
    :param lr_params: parameters for training
    :return:
    '''
    def __init__(self, lr_params):
        self.lr_params = lr_params
        self.logreg = LogisticRegression(**self.lr_params)

    def train(self, X, y):
        self.logreg.fit(X, y)

    def predict(self, X):
        mode = self.logreg.predict(X)
        return mode

    def predict_f(self, X):
        mode_prob = self.logreg.predict_proba(X)
        return mode_prob


def mergeGaussians(g1, g2, w1, w2):
    '''
    Merges two Gaussian distribution into single Gaussian
    :param g1, g2: gaussian mean and covariance as a flatten vector
    :param w1, w2: weights used for averaging the gaussians
    :return: flatten merged Gaussian
    '''
    dim = int(0.5 * (-1 + sqrt(1 + 4 * len(g1))))
    mu1 = g1[0:dim]
    mu2 = g2[0:dim]
    cov1 = np.reshape(g1[dim:], (-1, dim))
    cov2 = np.reshape(g2[dim:], (-1, dim))
    mu = w1*mu1 + w2*mu2
    # For un-normalized weights
    # cov = (w1*n1*cov1 + w2*n2*cov2 + np.matmul((w1*mu1 + w2*mu2), np.transpose((w1*mu1 + w2*mu2))))/((w1+w2)*(w1+w2)) - np.matmul(mu, np.transpose(mu))
    # For normalized weights
    cov = w1*cov1 + w2*cov2 + w1*np.matmul(mu1, np.transpose(mu1)) + w2*np.matmul(mu2, np.transpose(mu2)) - np.matmul(mu, np.transpose(mu))
    return np.append(mu, cov)


def scale(X, x_min, x_max):
    '''
    Performs standardization of vector
    :param X: data vector
    :param x_min, x_max: min and max range to scale
    :return: scaled X
    '''

    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


def computeDistance(f1, f2):
    '''
    Helper function for computing symmetric Bhattacharya distance between two Gaussian distribtion
    :param f1, f2:
    :return: Bhattacharya distance
    '''
    dim = int(0.5 * (-1 + sqrt(1 + 4 * len(f1))))
    mf1 = f1[0:dim]
    mf2 = f2[0:dim]
    covf1 = np.reshape(f1[dim:], (-1, dim))
    covf2 = np.reshape(f2[dim:], (-1, dim))
    return .5 * (bhattacharyyaGaussian(mf1, covf1, mf2, covf2) + bhattacharyyaGaussian(mf2, covf2, mf1, covf1))


def bhattacharyyaGaussian(pm, pv, qm, qv):
    '''
    Computes Bhattacharyya value between two Gaussians
    :param pm, pv: mean and covariance of first Gaussian
    :param qm, qv: mean and covariance of second Gaussian
    :return: Bhattacharya value
    :Author - TODO: ADD
    '''
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
    norm = 0.5 * np.log(ldpqv/(np.sqrt(ldpv*ldqv)))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    temp = np.matmul(diff.transpose(), np.linalg.pinv(pqv))
    dist = 0.125 * np.matmul(temp, diff)
    return np.float(dist + norm)


def fitGaussianDistribution(traj, features, transitions, minLength, gaussianEps):
    '''
    Fits Multivariate Gaussian distribution to a segment of trajectory
    :param traj: states from a trajectory statement (e.g position/velocity)
    :param feature: additional features to be used (e.g actions, contact force)
    :param transitions: transtions detected in the trajectory
    :param minLength: parameter used to check sufficient length of trajectory
    :param gaussianEps: eps parameter added along the diagonal to avoid singular Gaussian covariance
    :return: flatten Gaussian mean and covariance
    '''
    nseg = len(transitions)
    dynamicMat = []
    selectedSeg = []
    for k in range(0, nseg - 1):
        # Ensuring at least minLength samples is there between two transition point
        if (transitions[k + 1] - transitions[k]) > minLength:
            # x_t_1 = traj[(transitions[k] + 1):(transitions[k + 1] + 1), :]
            x_t = traj[transitions[k]:transitions[k + 1], :]
            f_t = features[transitions[k]:transitions[k + 1], :]
            feature_vect = np.hstack((x_t, f_t))
            meanGaussian = np.mean(feature_vect, axis=0)
            covGaussian = np.cov(feature_vect, rowvar=0)
            covGaussian = covGaussian + np.identity(covGaussian.shape[0], dtype=float)*gaussianEps
            # Check for singular matrics
            assert np.linalg.cond(covGaussian) < 1 / sys.float_info.epsilon

            selectedSeg.append(np.array([transitions[k], transitions[k + 1]]))
            dynamicMat.append(np.append(meanGaussian, covGaussian))

    return np.array(dynamicMat), np.array(selectedSeg)


def smoothing(indices):
    '''
    Helper function to remove redundant transition points
    :param indices: transition indices
    :return: filtered transition indices
    '''
    newIndices = indices
    for i in range(1, len(indices) - 1):
        if indices[i] != indices[i - 1] and indices[i] != indices[i + 1] and indices[i + 1] == indices[i - 1]:
            newIndices[i] = indices[i + 1]

    return newIndices


def identifyTransitions(traj, window_size, weight_prior, n_components):
    '''
    Transition detection function based on DPGMM and windowing approach
    :param traj: trajectory with states, action, contact forces
    :param window_size: windows size used to accumulate states
    :param weight_prior, n_components: parameter used for DPGMM
    :return: transition points (index) in the trajectory
    '''
    total_size = traj.shape[0]
    dim = traj.shape[1]
    demo_data_array = np.zeros((total_size - window_size, dim * window_size))
    inc = 0
    for i in range(window_size, total_size):
        window = traj[i - window_size:i, :]
        demo_data_array[inc, :] = np.reshape(window, (1, dim * window_size))
        inc = inc + 1

    estimator = BayesianGaussianMixture(n_components=n_components, n_init=10, max_iter=300, weight_concentration_prior=weight_prior, init_params='random', verbose=False)
    labels = estimator.fit_predict(demo_data_array)
    filtabels = smoothing(labels)
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
