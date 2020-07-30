import tensorflow.compat.v1 as tf1
from common.dataset import Dataset
import logger
import common.tf_util as U
import numpy as np
from common.mpi_adam import MpiAdam
from common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from common.math_util import zipsame, flatten_lists
from common.model_learning_utils import compute_likelihood
import time
import pickle
from model_learning import partialHybridModel
import option_critic_model
import difflib
import math
import gym
import pybullet_envs

tf1.disable_v2_behavior()

model_learning_params = {
    'per_train': 1,  # percentage of total rollouts to be trained
    'window_size': 2,  # window size of transition point clustering
    'weight_prior': 0.01,  # weight prior of DPGMM clustering for transition point
    'DBeps': 3.0,  # DBSCAN noise parameter for clustering segments
    'DBmin_samples': 2,  # DBSCAN minimum cluster size parameter for clustering segments
    'n_components': 2,  # number of DPGMM components to be used
    'minLength': 3,  # minimum segment length for Gaussian modelling
    'guassianEps': 1e-6,  # epsilon term added in Gaussian covariance
    'queueSize': 5000  # buffer size of samples
}
svm_grid_params = {
    'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=4, base=2.),
                   "gamma": np.logspace(-10, 10, endpoint=True, num=4, base=2.)},
    'scoring': 'accuracy',
    # 'cv': 5,
    'n_jobs': 4,
    'iid': False,
    'cv': 3,
}
svm_params_interest = {
    'kernel': 'rbf',
    'decision_function_shape': 'ovr',
    'tol': 1e-06,
    'probability': True,
}
svm_params_guard = {
    'kernel': 'rbf',
    'decision_function_shape': 'ovr',
    'tol': 1e-06,
    'probability': True,
}
from common.model_learning_utils import *
from hier_runner import add_vtarg_and_adv
'''
def sample_trajectory(pi, model, env, iteration, horizon=150, rolloutSize=50, render=False):
    """
            Generates rollouts for policy optimization
    """
    if render:
        env.setRender(True)
    else:
        env.setRender(False)

    #if iteration > 20:
    #    constraint = False
    #else:
    #    constraint = True

    constraint = False
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    num_options = pi.num_options
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...
    batch_size = int(horizon * rolloutSize)
    # Initialise history of arrays
    obs = np.array([ob for _ in range(batch_size)])
    rews = np.zeros(batch_size, 'float32')
    news = np.zeros(batch_size, 'int32')
    opts = np.zeros(batch_size, 'int32')
    activated_options = np.zeros((batch_size, num_options), 'float32')

    acs = np.array([ac for _ in range(batch_size)])
    model.currentMode = 0
    option, active_options_t = pi.get_option(ob, constraint)

    opt_duration = [[] for _ in range(num_options)]
    sample_index = 0
    curr_opt_duration = 0

    success = 0
    successFlag = False
    while sample_index < batch_size:
        ac = pi.act(True, ob, option)
        obs[sample_index] = ob
        news[sample_index] = new
        opts[sample_index] = option
        acs[sample_index] = ac
        activated_options[sample_index] = active_options_t

        # Step in the environment
        ob, rew, new, _ = env.step(ac)
        if math.isnan(ob[0]) or math.isnan(ob[1]):
            print("NAN value in observation. !!! check check")
            print("Force applied: ", ac)
            break

        rews[sample_index] = rew
        curr_opt_duration += 1
        # check if current option is about to end in this state
        nbeta = pi.get_tpred(ob)
        tprob = nbeta[option]

        if render:
            env.render()

        # Check for termination
        if tprob >= pi.term_prob:
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            model.currentMode = model.getNextMode(ob)
            option, active_options_t = pi.get_option(ob, constraint)

        cur_ep_ret += rew
        cur_ep_len += 1
        dist = env.getGoalDist()

        if np.linalg.norm(dist) < 0.05 and not successFlag:
            success = success + 1
            successFlag = True

        sample_index += 1

        if new or (sample_index > 0 and sample_index % horizon == 0):
            render = False
            env.setRender(False)
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            option, active_options_t = pi.get_option(ob, constraint)
            successFlag = False
            new = True

    env.close()
    print("Selected options")
    for o in range(0, num_options):
        print("Option: ", o, " - ", sum(opt_duration[o]))

    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    rollouts = {"ob": obs, "rew": rews, "new": news, "ac": acs, "opts": opts, "ep_rets": ep_rets, "ep_lens": ep_lens, "opt_dur": opt_duration, "activated_options": activated_options, "success": success}

    return rollouts


def add_vtarg_and_adv(rollouts, pi, gamma, lam, num_options):
    """
        Compute advantage and other value functions using GAE
    """
    obs = rollouts['seg_obs']
    # opts = rollouts['seg_opts']
    des_opts = rollouts['des_opts']
    des_opts = des_opts.astype(int)

    betas = []
    vpreds = []
    op_vpreds = []
    u_sws = []

    for sample in range(0, len(obs)):
        beta, vpred, op_vpred = pi.get_preds(obs[sample, :], False)
        vpred = np.squeeze(vpred)
        u_sw = (1 - beta) * vpred + beta * op_vpred
        betas.append(beta)
        vpreds.append(vpred)
        op_vpreds.append(op_vpred)
        u_sws.append(u_sw)

    u_sws = np.array(u_sws)

    new = np.append(rollouts["seg_news"], True)
    T = len(rollouts["seg_rews"])
    rew = rollouts["seg_rews"]

    gaelam = np.empty((num_options, T), 'float32')
    for des_opt in range(num_options):
        vpred_opt = u_sws[:, des_opt]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            if t == T - 1:
                delta = rew[t]
            else:
                delta = rew[t] + gamma * vpred_opt[t + 1] * nonterminal - vpred_opt[t]

            gaelam[des_opt, t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    rollouts["adv"] = gaelam.T[range(len(des_opts)), des_opts]
    rollouts["tdlamret"] = rollouts["adv"] + u_sws[range(len(des_opts)), des_opts]
    rollouts["is_adv"] = rollouts["adv"] * rollouts['is']
    rollouts["betas"] = np.array(betas)
    rollouts["vpreds"] = np.array(vpreds)
    rollouts["op_vpred"] = np.array(op_vpreds)
'''


def policy_fn(name, ob_space, ac_space, hybrid_model, num_options):
    return option_critic_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=[35, 25],
                                         model=hybrid_model, num_options=num_options, num_hid_layers=[2, 2],
                                         term_prob=0.5, eps=0.2)


f = open("results/MOAC/isexp2_5/data/rollout_data.pkl", "rb")
p = pickle.load(f)

horizon = 150
rolloutSize = 75
modes = 3
num_options = 9
queueSize = 5000
env = gym.make('BlockSlide2D-v1')
clip_param = 0.2
ent_coeff = 0.02  # clipping parameter epsilon, entropy coeff
optim_epochs = 20
optim_stepsize = 3e-4
optim_batchsize = 100  # optimization hypers
gamma = 0.99
lam = 0.95  # advantage estimation
max_iters = 0  # time constraint
adam_epsilon = 1.2e-4
schedule = 'linear'  # annealing for stepsize parameters (epsilon and adam)

ob_space = env.observation_space
ac_space = env.action_space

model = partialHybridModel(env, model_learning_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, num_options, rolloutSize)
pi = policy_fn("pi", ob_space, ac_space, model, num_options)  # Construct network for new policy
sampled_pi = policy_fn("sampled_pi", ob_space, ac_space, model, num_options)  # Network for sampled policy
assign_old_eq_new = U.function([], [], updates=[tf1.assign(oldv, newv) for (oldv, newv) in zipsame(sampled_pi.get_variables(), pi.get_variables())])
var_list = pi.get_trainable_variables()
adam = MpiAdam(var_list, epsilon=adam_epsilon)

U.initialize()
adam.sync()
i = 1
data = p[i]
rollouts = data['rollouts']
model.learnPreDefModes(rollouts)
model.learnTranstionRelation(rollouts, pi)
print(model.des_opts_s)
print("Model graph:", model.transitionGraph.nodes)
print("Model options:", model.transitionGraph.edges)
edges = list(model.transitionGraph.edges)
for i in range(0, len(edges)):
    print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])

'''
#rollouts = sample_trajectory(pi, model, env, iters_so_far, horizon=horizon, rolloutSize=rolloutSize, render=False)
model.updateModel(rollouts, sampled_pi)
datas = [0 for _ in range(num_options)]
add_vtarg_and_adv(rollouts, sampled_pi, gamma, lam, num_options)
ob, ac, opts, des_opts, atarg, tdlamret = rollouts["seg_obs"], rollouts["seg_acs"], rollouts["seg_opts"], rollouts["des_opts"], rollouts["is_adv"], rollouts["tdlamret"]

atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
if hasattr(pi, "ob_rms"):
    sampled_pi.ob_rms.update(ob)  # update running mean/std for policy
assign_old_eq_new() #sampled pi -> pi

datas = [0 for _ in range(num_options)]

sess = U.get_session()

for des_opt in range(num_options):
    indices = np.where(des_opts == des_opt)[0]
    datas[des_opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], opt=opts[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)
    optim_batchsize_corrected = optim_batchsize
    optim_epochs_corrected = np.clip(np.int(indices.size / optim_batchsize_corrected), 1, optim_epochs)
    print("Optim Epochs:", optim_epochs_corrected)
    logger.log("Optimizing...")
    for _ in range(optim_epochs_corrected):
        losses = []
        for batch in d.iterate_once(optim_batchsize_corrected):
            actual_mean, actual_std = sess.run([pi.pd.mode(), pi.pd.variance()], feed_dict={pi.ob: batch["ob"], pi.option:batch["seg_opts"]})

'''