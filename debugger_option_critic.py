import tensorflow.compat.v1 as tf1
from common.clustering_utils import hybridSegmentClustering
import option_critic_model
import gym
from common.dataset import Dataset
import logger
import common.tf_util as U
import numpy as np
from common.mpi_adam import MpiAdam
from common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from common.console_util import fmt_row
from common.math_util import explained_variance
import time
import pickle


clustering_params = {
    'per_train': 0.2,  # percentage of total rollouts to be trained
    'window_size': 2, # window size of transition point clustering
    'weight_prior': 0.05, # weight prior of DPGMM clustering for transition point
    'DBeps': 3.0,   # DBSCAN noise parameter for clustering segments
    'DBmin_samples': 2   # DBSCAN minimum cluster size parameter for clustering segments
}
svm_grid_params = {
    'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=11, base=2.),
                   "gamma": np.logspace(-10, 10, endpoint=True, num=11, base=2.)},
    'scoring': 'accuracy',
    # 'cv': 5,
    'n_jobs': -1,
    'iid': False,
    'cv': 3,
    }

svm_params_interest = {
    'kernel': 'rbf',
    'decision_function_shape': 'ovr',
    'tol': 1e-06,
    'probability': True,
    'C': 0.015625,
    'gamma': 1024.0
    }

svm_params_guard = {
    'kernel': 'rbf',
    'decision_function_shape': 'ovr',
    'tol': 1e-06,
    'probability': False,
    'C': 0.25,
    'gamma': 16.0
    }
lr_params_interest = {'C': 1, 'penalty': 'l2'}
lr_params_guard = {'C': 1, 'penalty': 'l2'}

tf1.disable_v2_behavior()

# collect trajectory
def sample_trajectory(pi, env, horizon=150, batch_size=12000):
    render = False
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    num_options = 2
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialise history of arrays
    obs = np.array([ob for _ in range(batch_size)])
    rews = np.zeros(batch_size, 'float32')
    news = np.zeros(batch_size, 'int32')
    opts = np.zeros(batch_size, 'int32')
    activated_options = np.zeros((batch_size, num_options), 'float32')

    last_options = np.zeros(batch_size, 'int32')
    acs = np.array([ac for _ in range(batch_size)])
    prevacs = acs.copy()

    option, active_options_t = pi.get_option(ob)
    last_option = option

    betas = []
    vpreds = []
    op_vpreds = []
    int_fcs = []
    op_probs = []

    ep_states = [[] for _ in range(num_options)]
    ep_states[option].append(ob)
    ep_num = 0

    opt_duration = [[] for _ in range(num_options)]
    t = 0
    i = 0
    curr_opt_duration = 0

    rollouts=[]
    observations = np.array([ob for _ in range(horizon)])
    actions = np.array([ac for _ in range(horizon)])
    while t < batch_size:
        prevac = ac
        ac = pi.act(True, ob, option)
        obs[t] = ob
        last_options[t] = last_option

        news[t] = new
        opts[t] = option
        acs[t] = ac
        prevacs[t] = prevac
        beta, vpred, op_vpred, op_prob, int_fc = pi.get_preds(ob)
        betas.append(beta)
        vpreds.append(vpred)
        op_vpreds.append(op_vpred)
        int_fcs.append(int_fc)
        op_probs.append(op_prob)

        activated_options[t] = active_options_t
        observations[i] = ob
        actions[i] = ac
        i=i+1

        if i==horizon:
            data = {'observations': observations, 'actions': actions}
            rollouts.append(data)
            i=0

        ob, rew, new, _ = env.step(ac)
        if render:
            env.render()
        rews[t] = rew

        curr_opt_duration += 1
        # check if current option is about to end in this state
        #print(beta)
        tprob = beta[option][0]
        if tprob > 0.75:
            term = True
        else:
            term = False

        if term:
            #print("Current Option Terminated !! ")
            #print("Sample number", t)
            #print("Next option", candidate_option)
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            last_option = option
            option, active_options_t = pi.get_option(ob)

        cur_ep_ret += rew
        cur_ep_len += 1

        if new or (t > 0 and t % horizon == 0):
            render = False
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ep_num += 1
            ob = env.reset()
            option, active_options_t = pi.get_option(ob)
            last_option = option


        t += 1

    betas = np.array(betas)
    vpreds = np.array(vpreds).reshape(batch_size, num_options)
    op_vpreds = np.squeeze(np.array(op_vpreds))
    op_probs = np.array(op_probs).reshape(batch_size, num_options)
    int_fcs = np.array(int_fcs).reshape(batch_size, num_options)
    last_betas = betas[range(len(last_options)), last_options]

    beta, vpred, op_vpred, op_prob, int_fc = pi.get_preds(ob)
    seg = {"ob": obs, "rew": rews, "vpred": np.array(vpreds), "op_vpred": np.array(op_vpreds), "new": news,
           "ac": acs, "opts": opts, "prevac": prevacs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new),
           "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': betas, 'next_term_p': beta,
           "opt_dur": opt_duration, "op_probs": np.array(op_probs), "last_betas": last_betas, "intfc": np.array(int_fcs),
           "activated_options": activated_options}

    return seg, rollouts


def policy_fn(name, ob_space, ac_space):
    return option_critic_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=2, num_options=num_options, dc=0, w_intfc=True, k=0)


U.make_session(num_cpu=1).__enter__()
env = gym.make('Block2D-v1')
env.seed(1)
U.initialize()


num_options = 2
optim_batchsize = 32
clip_param = 0.2
entcoeff = 0.0
optim_epochs = 10
adam_epsilon = 1e-5
lam = 0.95
gamma = 0.99
batch_size_per_episode=12000
cur_lrmult = 1.0
mainlr = 3e-4
np.random.seed(1)
tf1.set_random_seed(1)

ob_space = env.observation_space
ac_space = env.action_space
pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
pi.init_hybridmodel(lr_params_interest, lr_params_guard)
U.initialize()


p = pickle.load(open("option_critic_data_lr.pkl", "rb"))
data = p[35]
seg_adv = data['seg']
rollouts_adv = data['rollouts']
nmodes, segmentedRollouts, x_train, u_train, delx_train, label_train, label_t_train = hybridSegmentClustering(rollouts_adv, clustering_params)
pi.nmodes = nmodes
first_time = False
if nmodes > 1:
    label_train = np.squeeze(label_train)
    label_t_train = np.squeeze(label_t_train)
    xu_train = np.hstack((x_train, u_train))
    if first_time:
        pi.learn_hybridmodel(x_train, label_train, label_t_train)
        first_time = False
        print(pi.intfc.grid_results.best_params_)
        print(pi.termfc.grid_results.best_params_)
    else:
        pi.learn_hybridmodel(x_train, label_train, label_t_train)

stime = time.time()
seg, rollouts = sample_trajectory(pi, env, horizon=150, batch_size=batch_size_per_episode)
print("Collection time: ", time.time()-stime)

