import gym
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import common.tf_util as U
import numpy as np
from common.distribution import DiagGaussianPdType
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import sys
from sklearn.cluster import DBSCAN
from math import sqrt
from sklearn.svm import SVC
import gpflow
import time

from gpflow.utilities import print_summary

tf1.disable_v2_behavior()

# supressing warnings
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def scale(X, x_min, x_max):
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom


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

    print("[TSC] Discovered Transitions (number): ", len(transitions))
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
        if transitions[k + 1] - transitions[k] > 5:
            # ensuring at least one sample is there between two transition point
            x_t_1 = traj[(transitions[k] + 1):(transitions[k + 1] + 1), :]
            x_t = traj[transitions[k]:transitions[k + 1], :]
            u_t = action[transitions[k]:transitions[k + 1], :]
            feature_data_array = np.hstack((x_t, x_t_1))
            meanGaussian = np.mean(feature_data_array, axis=0)
            covGaussian = np.cov(feature_data_array, rowvar=0)
            covFeature = covGaussian.flatten()
            det = np.linalg.det(covGaussian)
            if np.linalg.cond(covGaussian) < 1 / sys.float_info.epsilon:
                # print("Segment Number: ", k)
                selectedSeg.append(np.array([transitions[k], transitions[k + 1]]))
                dynamicMat.append(np.append(meanGaussian, covGaussian))
            else:
                print("Singular Matrix !!! ")

    return np.array(dynamicMat), np.array(selectedSeg)


def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf1.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf1.matmul(x, w[option[0]])
    if bias:
        b = tf1.get_variable(name + "/b", [num_options, size], initializer=tf1.zeros_initializer())
        return ret + b[option[0]]

    else:
        return ret


start_time = time.time()
U.make_session(num_cpu=1).__enter__()

env = gym.make('Block2D-v1')
env.seed(1)
ob_space = env.observation_space
ac_space = env.action_space
sequence_length = None
num_hid_layers = 2
hid_size = 64
num_options = 2

# Creating the policy network
ob = U.get_placeholder(name="ob", dtype=tf1.float32, shape=[sequence_length] + list(ob_space.shape))
option = U.get_placeholder(name="option", dtype=tf1.int32, shape=[None])
pdtype = DiagGaussianPdType(ac_space.shape[0])

last_out = ob

# Value function
for i in range(num_hid_layers):
    last_out = tf1.nn.tanh(
        tf1.layers.dense(last_out, hid_size, name="vffc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
vpred = dense3D2(last_out, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:, 0]

# Termination condition
for i in range(num_hid_layers):
    last_out = tf1.nn.tanh(
        tf1.layers.dense(last_out, hid_size, name="termfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
tpred = tf1.nn.sigmoid(dense3D2(tf1.stop_gradient(last_out), 1, "termhead", option, num_options=num_options,
                                weight_init=U.normc_initializer(1.0)))[:, 0]
termination_sample = tf1.greater(tpred, tf1.random_uniform(shape=tf1.shape(tpred), maxval=1.))
print(termination_sample)

# Intra option policy
last_out = ob
for i in range(num_hid_layers):
    last_out = tf1.nn.tanh(
        tf1.layers.dense(last_out, hid_size, name="polfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))

mean = dense3D2(last_out, pdtype.param_shape()[0] // 2, "polfinal", option, num_options=num_options,
                weight_init=U.normc_initializer(0.01))
logstd = tf1.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0] // 2],
                          initializer=tf1.zeros_initializer())
pdparam = tf1.concat([mean, mean * 0.0 + logstd[option[0]]], axis=1)
pd = pdtype.pdfromflat(pdparam)
stochastic = tf1.placeholder(dtype=tf1.bool, shape=())
ac = U.switch(stochastic, pd.sample(), pd.mode())

# Interest Function
last_out = ob
for i in range(num_hid_layers):
    last_out = tf1.nn.tanh(
        tf1.layers.dense(last_out, hid_size, name="intfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
intfc = tf1.sigmoid(
    tf1.layers.dense(last_out, num_options, name="intfcfinal", kernel_initializer=U.normc_initializer(1.0)))

# Option policy
last_out = ob
for i in range(num_hid_layers):
    last_out = tf1.nn.tanh(
        tf1.layers.dense(last_out, hid_size, name="OP%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
op_pi = tf1.nn.softmax(
    tf1.layers.dense(last_out, num_options, name="OPfinal", kernel_initializer=U.normc_initializer(1.0)))

get_term = U.function([ob, option], [termination_sample])
act = U.function([stochastic, ob, option], [ac])
get_tpred = U.function([ob, option], [tpred])
get_vpred = U.function([ob, option], [vpred])
# get_intfc = U.function([ob], [intfc])
get_op = U.function([ob], [op_pi])


def get_intfc(ob):
    num = np.array(ob).shape[0]
    int_fc = np.array([[0.99, 0.1]])
    return np.repeat(int_fc, num, axis=0)


def get_act(stochastic, ob, option):
    ac1 = act(stochastic, ob[None], [option])
    return ac1[0]


def get_option(ob):
    op_prob = get_op([ob])
    int_func = get_intfc([ob])

    activated_options = []

    for int_val in int_func[0]:
        if int_val >= 0.5:
            activated_options.append(1.)
        else:
            activated_options.append(0.)
    indices = (-int_func[0]).argsort()[:2]
    if 1. not in activated_options:
        for i in indices:
            activated_options[i] = 1.

    pi_I = op_prob[0] * (activated_options * int_func) / np.sum(op_prob[0] * (activated_options * int_func), axis=1)

    return np.random.choice(range(len(op_prob[0][0])), p=pi_I[0]), activated_options


def get_allvpreds(obs, ob):
    obs = np.vstack((obs, ob[None]))

    # Get Q(s,w)
    vals = []
    for opt in range(num_options):
        vals.append(get_vpred(obs, [opt])[0])
    vals = np.array(vals).T

    op_prob = get_op(obs)
    int_func = get_intfc(obs)
    op_prob = op_prob[0]

    pi_I = op_prob * int_func / np.sum(op_prob * int_func, axis=1)[:, None]
    op_vpred = np.sum((pi_I * vals), axis=1)  # Get V(s)

    return vals[:-1], op_vpred[:-1], vals[-1], op_vpred[-1], op_prob[:-1], int_func[:-1]


def get_alltpreds(obs, ob):
    obs = np.vstack((obs, ob[None]))

    # Get B(s,w)
    betas = []
    for opt in range(num_options):
        betas.append(get_tpred(obs, [opt])[0])
    betas = np.array(betas).T

    return betas[:-1], betas[-1]


# collect trajectory
def sample_trajectory(env):
    horizon = 100
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Intialise history of arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    activated_options = np.zeros((horizon, num_options), 'float32')

    last_options = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    option, active_options_t = get_option(ob)

    option_terms = []
    option_plots = []
    int_vals = []
    option_plots.append(option)
    last_option = option

    term_prob = get_tpred([ob], [option])[0][0]
    option_terms.append(term_prob)
    int_val = get_intfc([ob])[0]
    # print("Interest Value: ", int_val)
    # print("Termination probability", term_prob)
    int_vals.append(int_val[option])

    ep_states = [[] for _ in range(num_options)]
    ep_states[option].append(ob)
    ep_num = 0
    optpol_p = []

    value_val = []
    opt_duration = [[] for _ in range(num_options)]
    t = 0
    curr_opt_duration = 0

    while t < horizon:
        prevac = ac
        ac = get_act(True, ob, option)
        obs[t] = ob
        last_options[t] = last_option

        news[t] = new
        opts[t] = option
        acs[t] = ac
        prevacs[t] = prevac
        activated_options[t] = active_options_t

        ob, rew, new, _ = env.step(ac)
        # env.render()
        rews[t] = rew
        realrews[t] = rew
        option_plots.append(option)
        curr_opt_duration += 1
        term = get_term([ob], [option])[0][0]
        int_vals.append(term)

        candidate_option, active_options_t = get_option(ob)
        if term:
            # print("option terminated")
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            last_option = option
            option = candidate_option
            term_prob = get_tpred([ob], [option])[0][0]

        option_terms.append(term_prob)
        int_val = get_intfc([ob])
        ep_states[option].append(ob)

        ep_states[option].append(ob)

        cur_ep_ret += rew
        cur_ep_len += 1

        if new:
            print("Inserted !! ")
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0

            ep_num += 1
            ob = env.reset()
            option, active_options_t = get_option(ob)
            term_prob = get_tpred([ob], [option])[0][0]
            last_option = option
            ep_states[option].append(ob)

        t += 1

    vpreds, op_vpreds, vpred, op_vpred, op_probs, intfc = get_allvpreds(obs, ob)
    term_ps, term_p = get_alltpreds(obs, ob)
    last_betas = term_ps[range(len(last_options)), last_options]

    # print(opt_duration)
    # plt.subplot(1, 4, 1)
    # plt.plot(obs[:, 0])
    # plt.subplot(1, 4, 2)
    # plt.plot(option_plots)
    # plt.subplot(1, 4, 3)
    # plt.plot(term_ps)

    seg = {"ob": obs, "rew": rews, "realrew": realrews, "vpred": vpreds, "op_vpred": op_vpreds, "new": news,
           "ac": acs, "opts": opts, "prevac": prevacs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new),
           "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': term_ps, 'next_term_p': term_p,
           "opt_dur": opt_duration, "op_probs": op_probs, "last_betas": last_betas, "intfc": intfc,
           "activated_options": activated_options}

    return seg


def compute_advantage(seg):
    # Compute advantage and other value functions using GAE
    lam = 0.95
    gamma = 0.99

    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    op_vpred = np.append(seg["op_vpred"], seg["nextop_vpred"])
    T = len(seg["rew"])
    seg["op_adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0

    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        # TD error
        delta = rew[t] + gamma * op_vpred[t + 1] * nonterminal - op_vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    # above equations not used

    term_p = np.vstack((np.array(seg["term_p"]), np.array(seg["next_term_p"])))
    q_sw = np.vstack((seg["vpred"], seg["nextvpred"]))
    # Utility function in option framework
    u_sw = (1 - term_p) * q_sw + term_p * np.tile(op_vpred[:, None], num_options)

    opts = seg["opts"]
    new = np.append(seg["new"], 0)
    T = len(seg["rew"])
    rew = seg["rew"]
    gaelam = np.empty((num_options, T), 'float32')
    for opt in range(num_options):
        vpred = u_sw[:, opt]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[opt, t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["adv"] = gaelam.T[range(len(opts)), opts]
    seg["tdlamret"] = seg["adv"] + u_sw[range(len(opts)), opts]


def getSeg(l, trajMat):
    count = 0
    for i in range(0, trajMat.shape[0]):
        for j in range(0, trajMat[i][1].shape[0]):
            if count == l:
                return i, j
            count = count + 1

    print("Error: Did not get segment !! ")
    return None


# run training
U.initialize()
# Hyperparameters
window_size = 2
weight_prior = 0.05
DBeps = 3.0
DBmin_samples = 2
totalIter = 20
num_rollouts = 100
num_options = 2
rollout = 0
rows = 5
cols = 10

trajMat = []
p = []
currIter = 0
optim_batchsize = 32

while currIter < totalIter:
    currIter += 1

    while rollout < num_rollouts:
        seg = sample_trajectory(env)
        p.append(seg)
        # clustering results
        states = seg['ob']
        action = seg['ac']
        tp = identifyTransitions(states, window_size, weight_prior)
        '''
        plt.subplot(rows, cols, rollout + 1)
        plt.plot(states[:, 0], 'r')
        for i in range(0, len(tp)):
            point = states[tp[i], 0]
            plt.subplot(rows, cols, rollout + 1)
            plt.plot(tp[i], point, 'bo-')
        '''
        # fitting Gaussian Dynamic model
        fittedModel, selTraj = fitGaussianDistribution(states[:, 0:2], action, tp)
        trajMat.append(np.array([rollout, selTraj]))
        if rollout == 0:
            dynamicMat = fittedModel
        else:
            dynamicMat = np.concatenate((dynamicMat, fittedModel), axis=0)

        rollout += 1

    trajMat = np.array(trajMat)
    print(np.array(dynamicMat).shape)

    # DBSCAN based clustering
    db = DBSCAN(eps=DBeps, min_samples=DBmin_samples, metric=computeDistance)
    labels = db.fit_predict(dynamicMat)
    print(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    '''
    segCount = 0
    for rollout in range(0, num_rollouts):
        traj = p[rollout]
        states = traj['ob']
        plt.subplot(rows, cols, num_rollouts + rollout + 1)
        for segment in range(0, trajMat[rollout][1].shape[0]):
            segTime = np.arange(trajMat[rollout][1][segment][0], (trajMat[rollout][1][segment][1] + 1))
            segTraj = states[trajMat[rollout][1][segment][0]:(trajMat[rollout][1][segment][1] + 1), 0]
            if labels[segCount] == 0:
                plt.plot(segTime, segTraj, 'r')
            elif labels[segCount] == 1:
                plt.plot(segTime, segTraj, 'g')
            elif labels[segCount] == 2:
                plt.plot(segTime, segTraj, 'b')
            else:
                plt.plot(segTime, segTraj, 'k')
            segCount = segCount + 1

    plt.savefig("mygraph.png")
    '''

    segCount = 0
    for rollout in range(0, num_rollouts):
        traj = p[rollout]
        states = traj['ob']
        input = traj['ac']
        delta_traj = []
        for t in range(len(states) - 1):
            delta_states = states[t + 1, :] - states[t, :]
            delta_traj.append(delta_states)
        delta_traj.append(delta_states)
        delta_traj = np.array(delta_traj)
        for segment in range(0, trajMat[rollout][1].shape[0]):
            segTraj = states[trajMat[rollout][1][segment][0]:(trajMat[rollout][1][segment][1] + 1), :]
            segAction = input[trajMat[rollout][1][segment][0]:(trajMat[rollout][1][segment][1] + 1), :]
            segDelta = delta_traj[trajMat[rollout][1][segment][0]:(trajMat[rollout][1][segment][1] + 1), :]
            if labels[segCount] >= 0:
                if segment == 0:
                    segLabel = labels[segCount] * np.ones((len(segTraj), 1))
                else:
                    segLabel = np.vstack((segLabel, labels[segCount] * np.ones((len(segTraj), 1))))

                if segCount == 0:
                    x_data = segTraj
                    u_data = segAction
                    delta_data = segDelta
                    label_data = labels[segCount] * np.ones((len(segTraj), 1))

                else:
                    x_data = np.vstack((x_data, segTraj))
                    u_data = np.vstack((u_data, segAction))
                    delta_data = np.vstack((delta_data, delta_traj))
                    label_data = np.vstack((label_data, labels[segCount] * np.ones((len(segTraj), 1))))

            segCount = segCount + 1

        segLabel_t_data = segLabel
        for i in range(len(segLabel) - 1):
            segLabel_t_data[i] = segLabel[i + 1]
        if rollout == 0:
            label_t_data = segLabel_t_data
        else:
            label_t_data = np.vstack((label_t_data, segLabel_t_data))

    print(x_data.shape)
    print(label_data.shape)
    print(label_t_data.shape)
    print('Done')

    # Generating interest function
    if (n_clusters_ > 1):
        # plt.subplot(rows, cols, rollouts + 3)
        # plt.plot(label_data)
        clf = SVC()
        svm_grid_params = {
            'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=11, base=2.),
                           "gamma": np.logspace(-10, 10, endpoint=True, num=11, base=2.)},
            'scoring': 'accuracy',
            # 'cv': 5,
            'n_jobs': -1,
            'iid': False,
            'cv': 3,
        }
        svm_params = {
            'kernel': 'rbf',
            'decision_function_shape': 'ovr',
            'tol': 1e-06,
            'probability': True
        }
        clf = SVC(**svm_params)
        clf.fit(x_data, np.squeeze(label_data))
        # pred_label = clf.predict(x_data[650:700, :])
        # pred_prob = clf.predict_proba((x_data[650:700, :]))
        # print(pred_label)
        # print(pred_prob)
    else:
        print('Number of clusters less')
    '''
    # model learning
    for i in range(0, n_clusters_):
        indices = np.where(label_data == i)
        x = x_data[indices[0], :]
        u = u_data[indices[0], :]
        del_y = delta_data[indices[0], :]
        dim = x.shape[1]
        print(x.shape)
        print(u.shape)
        print(del_y.shape)
        for j in range(0, dim):
            print(j)
            del_y_train = del_y[:, i]
            del_y_train = np.expand_dims(del_y_train, axis=1)
            xu_train = np.hstack((x, u))
            liklihood_variance_init = 0.001
            k = gpflow.kernels.SquaredExponential(xu_train.shape[1])
            m = gpflow.models.GPR(data=(xu_train, del_y_train), kernel=k)
            gpflow.utilities.set_trainable(m.likelihood.variance, False)
            m.likelihood.variance.assign(liklihood_variance_init)
            #print_summary(m)
            # m.kern.lengthscales.assign(lengthscale_init)
            # m.kern.variance.assign(float(variance_init))
            # print(m.as_pandas_table())
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
            #print_summary(m)

    delta_time = time.time()-start_time
    print("Total time for training model : ", delta_time)
    '''
