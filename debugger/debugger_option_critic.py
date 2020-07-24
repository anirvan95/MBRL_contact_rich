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

    acs = np.array([ac for _ in range(batch_size)])
    option, active_options_t = pi.get_option(ob)

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
        if render:
            env.render()

        rews[sample_index] = rew
        curr_opt_duration += 1

        # check if current option is about to end in this state
        nbeta = pi.get_tpred(ob)
        tprob = nbeta[option]
        if tprob >= pi.term_prob:
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            model.currentMode = model.getNextMode(ob)
            option, active_options_t = pi.get_option(ob)

        cur_ep_ret += rew
        cur_ep_len += 1
        dist = env.getGoalDist()

        if np.linalg.norm(dist) < 0.025 and not successFlag:
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
            option, active_options_t = pi.get_option(ob)
            successFlag = False
            new = True

    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    seg = {"ob": obs, "rew": rews, "new": news, "ac": acs, "opts": opts, "ep_rets": ep_rets, "ep_lens": ep_lens, "opt_dur": opt_duration, "activated_options": activated_options, "success": success}

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
    obs = rollouts['seg_obs']
    opts = rollouts['seg_opts']
    des_opts = rollouts['des_opts']
    des_opts = des_opts.astype(int)

    betas = []
    vpreds = []
    op_vpreds = []
    u_sws = []

    for sample in range(0, len(obs)):
        beta, vpred, op_vpred = pi.get_preds(obs[sample, :])
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
    print(rollouts["adv"].shape)
    print(rollouts["is"].shape)
    rollouts["is_adv"] = rollouts["adv"] * rollouts['is']
    rollouts["betas"] = np.array(betas)
    rollouts["vpreds"] = np.array(vpreds)
    rollouts["op_vpred"] = np.array(op_vpreds)


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
#epochNum = 35
#data = p[epochNum]
#rollouts = data['seg']

rollouts = sample_trajectory(pi, model, env, horizon=150, batch_size=12000, render=False)
model.learnPreDefModes(rollouts)
model.learnTranstionRelation(rollouts, pi)
model.learnGuardF()
model.learnModeF()
gamma = 0.99
lam = 0.95
add_vtarg_and_adv(rollouts, gamma, lam, num_options)
print("Done in : ", time.time()-stime)

edges = list(model.transitionGraph.edges)
for i in range(0, len(edges)):
    print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])
print("Options: ", model.nOptions)

data = {'seg': rollouts}
pickle.dump(data, open("rollouts_test.pkl", "wb"))

wb = Workbook()
sheet = wb.add_sheet('Final Data check')
obs = rollouts['seg_obs']
acs = rollouts['seg_acs']
print("Action dimenstion", acs.shape)
imp_s = rollouts['is']
adv = rollouts['is_adv']
print(adv.shape)
tdlamret = rollouts['tdlamret']
des_op = rollouts['des_opts']
op_vpred = rollouts['op_vpred']
rews = rollouts['seg_rews']
opts = rollouts['seg_opts']
betas = rollouts['betas']
for i in range(0, len(obs)):
    sheet.write(i, 0, str(obs[i, 0]))
    sheet.write(i, 1, str(imp_s[i]))
    sheet.write(i, 2, str(rews[i]))
    sheet.write(i, 3, str(adv[i]))
    sheet.write(i, 4, str(tdlamret[i]))
    sheet.write(i, 5, str(opts[i]))
    sheet.write(i, 6, str(des_op[i]))
    sheet.write(i, 7, str(betas[i, 0]))



wb.save('advantage_check.xls')