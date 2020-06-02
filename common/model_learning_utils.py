import numpy as np
import gpflow
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time


class multiDimGaussianProcess():
    def __init__(self, gpr_params):
        self.gp_param = gpr_params

    def fit(self, X, Y):
        assert(Y.shape[1]>=2)
        assert (X.shape[1]>=2)
        self.gp_list = []
        in_dim = X.shape[1]
        out_dim = Y.shape[1]
        self.out_dim = out_dim

        # Possible parallelization here
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
    def __init__(self, svm_params):
        self.svm_params = svm_params
        self.clf = SVC(**self.svm_params)

    def train(self, X, y):
        '''
        Trains SVMs for interest and guard functions
        :param svm_grid_params:
        :param svm_params:
        :param XU_t:
        :param labels_t:
        :return:
        '''
        self.clf.fit(X, y)

    def predict(self, X):
        mode = self.clf.predict(X)
        return mode

    def predict_f(self, X):
        mode_prob = self.clf.predict_proba(X)
        return mode_prob


class LRPrediction(object):
    def __init__(self, lr_params):
        self.lr_params = lr_params
        self.logreg = LogisticRegression(**self.lr_params)

    def train(self, X, y):
        '''
        Trains SVMs for interest and guard functions
        :param svm_grid_params:
        :param svm_params:
        :param XU_t:
        :param labels_t:
        :return:
        '''
        self.logreg.fit(X, y)

    def predict(self, X):
        mode = self.logreg.predict(X)
        return mode

    def predict_f(self, X):
        mode_prob = self.logreg.predict_proba(X)
        return mode_prob


def learnTransitionRelation(nmodes, segmentedRollouts):
    transitionRel = np.zeros((nmodes,nmodes))
    rollouts = len(segmentedRollouts)
    y_data = []
    x_data = []
    firstDataFlag = True
    for rollout in range(0, rollouts):
        for t in range(0, len(segmentedRollouts[rollout]['label'])):
            if segmentedRollouts[rollout]['label'][t] != segmentedRollouts[rollout]['label_t'][t]:
                # Updating the transition relations
                transitionRel[int(segmentedRollouts[rollout]['label'][t]), int(segmentedRollouts[rollout]['label_t'][t])] = 1
                if firstDataFlag:
                    y_data = segmentedRollouts[rollout]['x'][t+1]
                    x_data = np.hstack((segmentedRollouts[rollout]['x'][t], segmentedRollouts[rollout]['u'][t], segmentedRollouts[rollout]['label'][t], segmentedRollouts[rollout]['label_t'][t]))
                    firstDataFlag = False
                else:
                    y_data = np.vstack((y_data, segmentedRollouts[rollout]['x'][t+1]))
                    x_data = np.vstack((x_data, np.hstack((segmentedRollouts[rollout]['x'][t], segmentedRollouts[rollout]['u'][t], segmentedRollouts[rollout]['label'][t], segmentedRollouts[rollout]['label_t'][t]))))
    return y_data, x_data
