import tensorflow.compat.v1 as tf1
from model_learning import partialHybridModel
import option_critic_model
import gym
import pybullet_envs
import common.tf_util as U
import numpy as np
import time

tf1.disable_v2_behavior()

clustering_params = {
    'per_train': 0.5,  # percentage of total rollouts to be trained
    'window_size': 2,  # window size of transition point clustering
    'weight_prior': 0.01,  # weight prior of DPGMM clustering for transition point
    'DBeps': 3.0,  # DBSCAN noise parameter for clustering segments
    'DBmin_samples': 2,  # DBSCAN minimum cluster size parameter for clustering segments
    'n_components': 2,  # number of DPGMM components to be used
    'minLength': 8
}
svm_grid_params = {
    'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=11, base=2.),
                   "gamma": np.logspace(-10, 10, endpoint=True, num=11, base=2.)},
    'scoring': 'accuracy',
    # 'cv': 5,
    'n_jobs': 2,
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
        last_options[sample_index] = last_option
        news[sample_index] = new
        opts[sample_index] = option
        acs[sample_index] = ac
        prev_acs[sample_index] = prevac
        beta, vpred, op_vpred = pi.get_preds(ob)

        betas.append(beta[0])
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
        dist = ob[:3] - GOAL

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
            new_rollout = True
            new = True

        sample_index += 1

    betas = np.array(betas)
    vpreds = np.array(vpreds).reshape(batch_size, num_options)
    op_vpreds = np.squeeze(np.array(op_vpreds))
    last_betas = betas[range(len(last_options)), last_options]
    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    seg = {"ob": obs, "rew": rews, "vpred": np.array(vpreds), "op_vpred": np.array(op_vpreds), "new": news,
           "ac": acs, "opts": opts, "prevac": prev_acs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new), "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': betas,
           'next_term_p': beta[0], "last_betas": last_betas,
           "opt_dur": opt_duration, "activated_options": activated_options, "success": success}

    return seg


def policy_fn(name, ob_space, ac_space, hybrid_model, num_options):
    return option_critic_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=[32, 32, 16], model=hybrid_model, num_options=num_options, num_hid_layers=[2, 2, 2], term_prob=0.5, k=0.5, rg=10)


U.make_session(num_cpu=1).__enter__()
env = gym.make('Block2D-v2')
env.seed(1)
U.initialize()

num_options = 12
optim_batchsize = 32
clip_param = 0.2
entcoeff = 0.0
optim_epochs = 10
adam_epsilon = 1e-5
lam = 0.95
gamma = 0.99
batch_size_per_episode = 12000
cur_lrmult = 1.0
mainlr = 3e-4
horizon = 150
rolloutSize = 50
modes = 4
queueSize = 20000

np.random.seed(1)
tf1.set_random_seed(1)

ob_space = env.observation_space
ac_space = env.action_space

# Initialize the model

model = partialHybridModel(clustering_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, num_options, rolloutSize, queueSize)
pi = policy_fn("pi", ob_space, ac_space, model, num_options)  # Construct network for new policy
U.initialize()
stime = time.time()
seg = sample_trajectory(pi, model, env, horizon=150, batch_size=batch_size_per_episode)
#update policy

#update model
model.updateModel(seg)

print("Collection time: ", time.time() - stime)
