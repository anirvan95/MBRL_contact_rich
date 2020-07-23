import tensorflow.compat.v1 as tf1
import option_critic_model
import common.tf_util as U
from model_learning import partialHybridModel
import pickle
import numpy as np
import time
import gym
import pybullet_envs
import xlwt
from xlwt import Workbook
from common.model_learning_utils import *

tf1.disable_v2_behavior()

model_learning_params = {
    'per_train': 1,  # percentage of total rollouts to be trained
    'window_size': 2,  # window size of transition point clustering
    'weight_prior': 0.01,  # weight prior of DPGMM clustering for transition point
    'DBeps': 3.0,  # DBSCAN noise parameter for clustering segments
    'DBmin_samples': 2,  # DBSCAN minimum cluster size parameter for clustering segments
    'n_components': 2,  # number of DPGMM components to be used
    'minLength': 8,  # minimum segment length for Gaussian modelling
    'guassianEps': 1e-6,  # epsilon term added in Gaussian covariance
    'queueSize': 5000  # buffer size of samples
}
svm_grid_params = {
    'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=11, base=2.),
                   "gamma": np.logspace(-10, 10, endpoint=True, num=11, base=2.)},
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
    'C': 1024,
    'gamma': 16.0
}
svm_params_guard = {
    'kernel': 'rbf',
    'decision_function_shape': 'ovr',
    'tol': 1e-06,
    'probability': True,
    'C': 1024,
    'gamma': 4.0
}


# collect trajectory
def sample_trajectory(pi, model, env, horizon=150, batch_size=12000, render=False):
    """
            Generates rollouts for policy optimization
    """
    GOAL = np.array([0, 0.5])
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    num_options = pi.num_options
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
    prev_acs = acs.copy()
    option, active_options_t = pi.get_option(ob)
    last_option = option

    betas = []
    vpreds = []
    op_vpreds = []

    opt_duration = [[] for _ in range(num_options)]
    sample_index = 0
    curr_opt_duration = 0

    success = 0
    successFlag = False
    while sample_index < batch_size:
        prevac = ac
        ac = pi.act(True, ob, option)
        obs[sample_index] = ob
        last_options[sample_index] = int(last_option)
        news[sample_index] = new
        opts[sample_index] = option
        acs[sample_index] = ac
        prev_acs[sample_index] = prevac
        beta, vpred, op_vpred = pi.get_preds(ob)

        betas.append(beta)
        vpreds.append(vpred * (1 - new))
        op_vpreds.append(op_vpred)
        activated_options[sample_index] = active_options_t

        # Step in the environment
        ob, rew, new, _ = env.step(ac)
        if render:
            env.render()

        rews[sample_index] = rew
        curr_opt_duration += 1
        # check if current option is about to end in this state
        nbeta = pi.get_tpred(ob)
        tprob = nbeta[option]
        # termination =
        if tprob >= pi.term_prob:
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            last_option = option
            model.currentMode = model.getNextMode(ob)
            option, active_options_t = pi.get_option(ob)

        cur_ep_ret += rew
        cur_ep_len += 1
        dist = env.getGoalDist()

        if np.linalg.norm(dist) < 0.025 and not successFlag:
            success = success + 1
            successFlag = True

        if new or (sample_index > 0 and sample_index % horizon == 0):
            render = False
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            option, active_options_t = pi.get_option(ob)
            last_option = option
            successFlag = False
            new = True

        sample_index += 1

    betas = np.array(betas)
    print(betas)
    print(betas.shape)
    vpreds = np.array(vpreds).reshape(batch_size, num_options)
    op_vpreds = np.squeeze(np.array(op_vpreds))
    print(last_options)
    print(range(len(last_options)))
    last_betas = betas[range(len(last_options)), last_options]
    #TODO:add last_betas

    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    seg = {"ob": obs, "rew": rews, "vpred": np.array(vpreds), "op_vpred": np.array(op_vpreds), "new": news,
           "ac": acs, "opts": opts, "prevac": prev_acs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new), "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': betas,
           'next_term_p': beta[0], "opt_dur": opt_duration, "activated_options": activated_options, "success": success}

    return seg


def policy_fn(name, ob_space, ac_space, hybrid_model, num_options):
    return option_critic_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=[32, 32, 32],
                                         model=hybrid_model, num_options=num_options, num_hid_layers=[2, 2, 2],
                                         term_prob=0.5, eps=0.001)


def compute_likelihood(mean, std, ac):
    likelihood = (np.exp(-((ac[0] - mean[0]) * (ac[0] - mean[0])) / (2 * std[0] * std[0])) / (
        np.sqrt(2 * np.pi * std[0] * std[0]))) * (
                             np.exp(-((ac[1] - mean[1]) * (ac[1] - mean[1])) / (2 * std[1] * std[1])) / (
                         np.sqrt(2 * np.pi * std[1] * std[1])))
    return likelihood


def add_vtarg_and_adv(rollouts, gamma, lam, num_options):
    """
        Compute advantage and other value functions using GAE
    """
    obs = rollouts['ob']
    opts = rollouts['opts']
    obs_trimmed = obs[0:(horizon * (rolloutSize - 1)), :]
    des_opts = rollouts['des_opts']
    des_opts = des_opts.astype(int)

    betas = []
    vpreds = []
    op_vpreds = []
    u_sws = []

    for sample in range(0, len(obs_trimmed)):
        beta, vpred, op_vpred = pi.get_preds(obs_trimmed[sample, :])
        vpred = np.squeeze(vpred)
        u_sw = (1 - beta) * vpred + beta * op_vpred
        betas.append(beta)
        vpreds.append(vpred)
        op_vpreds.append(op_vpred)
        u_sws.append(u_sw)

    u_sws = np.array(u_sws)

    new = rollouts["new"]
    new_trimmed = new[0:(horizon * (rolloutSize - 1))]
    T = len(new_trimmed)
    new_trimmed = np.append(new_trimmed, 1)
    rew = rollouts["rew"]
    rew_trimmed = rew[0:(horizon * (rolloutSize - 1))]

    gaelam = np.empty((num_options, T), 'float32')
    for des_opt in range(num_options):
        vpred_opt = u_sws[:, des_opt]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new_trimmed[t + 1]
            if t == T - 1:
                delta = rew_trimmed[t]
            else:
                delta = rew_trimmed[t] + gamma * vpred_opt[t + 1] * nonterminal - vpred_opt[t]

            gaelam[des_opt, t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

        rollouts["adv"] = gaelam.T[range(len(des_opts)), des_opts]
        rollouts["tdlamret"] = rollouts["adv"] + u_sws[range(len(des_opts)), des_opts]
        rollouts["is_adv"] = np.dot(rollouts["adv"], rollouts['is'])



f = open("results/MOAC/exp_5/data/rollout_data.pkl", "rb")
p = pickle.load(f)
f.close()
horizon = 150
rolloutSize = 75
modes = 3
num_options = 9
queueSize = 5000
env = gym.make('BlockSlide2D-v1')
env.seed(1)

U.make_session(num_cpu=1).__enter__()

np.random.seed(1)
tf1.set_random_seed(1)

ob_space = env.observation_space
ac_space = env.action_space

# Initialize the model
model = partialHybridModel(env, model_learning_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, num_options, rolloutSize)
pi = policy_fn("pi", ob_space, ac_space, model, num_options)  # Construct network for new policy
policy_path = "results/MOAC/exp_5/model/"

U.initialize()

# Load trained model
#U.load_state(policy_path)

stime = time.time()
epochNum = 35
data = p[epochNum]
rollouts = data['seg']
model.learnPreDefModes(rollouts)
model.learnTranstionRelation(rollouts, pi)
model.learnGuardF()
model.learnModeF()


# compute advantage returns
gamma = 0.99
lam = 0.95

add_vtarg_and_adv(rollouts, gamma, lam, num_options)

print("Done in : ", time.time() - stime)
print("Done baby")

