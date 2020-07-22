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

wb = Workbook()
sheet = wb.add_sheet('Sheet 1')

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
# Importance Sampling test
data = p[epochNum]
rollouts = data['seg']
env_id = env.unwrapped.spec.id
train_rollouts = int(model_learning_params['per_train']) * rolloutSize
obs = rollouts['ob']
opts = rollouts['opts']
acs = rollouts['ac']
desOpts = opts
segmented_traj = []
seg_labels = []
for rollout in range(0, train_rollouts):
    states = obs[rollout * horizon:(horizon + rollout * horizon), :]
    traj_time = len(states)
    tp = []
    for t in range(0, traj_time - 1):
        label = obtainMode(env_id, states[t, :])
        label_t = obtainMode(env_id, states[t + 1, :])
        dataDict = {'x': states[t, :], 'label': label, 'label_t': label_t}
        model.dataset[int(label)].append(dataDict)
        if label != label_t:
            tp.append(t)
    tp.append(0)
    tp.append(traj_time)
    tp.sort()
    tp = np.array(tp)
    selectedSeg = []
    for k in range(0, len(tp) - 1):
        if (tp[k + 1] - tp[k]) > model_learning_params['minLength']:
            selectedSeg.append(np.array([tp[k], tp[k + 1]]))
            seg_labels.append(obtainSegMode(env_id, states[tp[k]:tp[k + 1], :]))
    segmented_traj.append(np.array([rollout, selectedSeg]))

labels = np.array(seg_labels)
segment_data = np.array(segmented_traj)
# print(labels)
# print(segment_data)
desOptionsLabels = np.zeros(labels.shape)
segCount = 0
IS = []
for rollout in range(0, train_rollouts-1):
    #print("Rollout: ", rollout)
    opt = opts[rollout * horizon:(horizon + rollout * horizon)]
    states = obs[rollout * horizon:(horizon + rollout * horizon), :]
    action = acs[rollout * horizon:(horizon + rollout * horizon), :]
    rolloutIndex = rollout * horizon
    numSegments = len(segment_data[rollout][1])
    for segment in range(0, numSegments):
        #print("Segment Num: ", segment)
        # Avoid Noisy segments
        if labels[segCount] >= 0:
            # Adding mode to GOAL edge e.g. 0 > goal, 1 > goal etc
            if not model.transitionGraph.has_edge(labels[segCount], 'goal'):
                model.nOptions += 1
                model.transitionGraph.add_weighted_edges_from([(labels[segCount], 'goal', model.nOptions)])

            if labels[segCount + 1] >= 0:  # Avoid noisy next segment
                # Adding mode to GOAL edge e.g. 1 > goal, 1 > goal etc
                if not model.transitionGraph.has_edge(labels[segCount + 1], 'goal'):
                    model.nOptions += 1
                    model.transitionGraph.add_weighted_edges_from([(labels[segCount + 1], 'goal', model.nOptions)])

                # Checking for transition detection while ignoring the last segment
                if labels[segCount] != labels[segCount + 1] and segment < numSegments - 1:
                    if not (model.transitionGraph.has_edge(labels[segCount], labels[segCount + 1])):
                        model.nOptions += 1
                        model.transitionGraph.add_weighted_edges_from([(labels[segCount], labels[segCount + 1], model.nOptions)])

                    #print(list(model.transitionGraph.edges))
                    # Assign the desired option for transition
                    # 0>1
                    if labels[segCount + 1] > labels[segCount]:
                        desOptionsLabels[segCount] = model.transitionGraph[labels[segCount]][labels[segCount + 1]]['weight']
                        desOpts[rolloutIndex + segment_data[rollout][1][segment][0]:rolloutIndex + segment_data[rollout][1][segment][1]] = model.transitionGraph[labels[segCount]][labels[segCount + 1]]['weight']
                    # 1>0 gets assigned to 1>goal
                    else:
                        desOptionsLabels[segCount] = model.transitionGraph[labels[segCount]]['goal']['weight']
                        desOpts[rolloutIndex + segment_data[rollout][1][segment][0]:rolloutIndex + segment_data[rollout][1][segment][1]] = model.transitionGraph[labels[segCount]]['goal']['weight']

                # Assigning desired option for last segment or only one segment
                else:
                    desOptionsLabels[segCount] = model.transitionGraph[labels[segCount]]['goal']['weight']
                    desOpts[rolloutIndex + segment_data[rollout][1][segment][0]:rolloutIndex + segment_data[rollout][1][segment][1]] = model.transitionGraph[labels[segCount]]['goal']['weight']

            segStates = states[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1] + 1), :]
            segAction = action[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1] + 1), :]
            is_ratio = 1
            for t in range(0, len(segStates)):
                desired_mean, desired_std = pi.get_ac_dist(segStates[t, :], desOptionsLabels[segCount])
                actual_mean, actual_std = pi.get_ac_dist(segStates[t, :], 0)
                is_ratio = is_ratio*compute_likelihood(desired_mean, desired_std, segAction[t, :])/compute_likelihood(actual_mean, actual_std, segAction[t, :])
                IS.append(is_ratio)
        xl_sheet_string = 'seg : ' + str(segment_data[rollout][1][segment]) + ', l :' + str(labels[segCount]) + ', do :' + str(desOptionsLabels[segCount])
        sheet.write(rollout, segment, xl_sheet_string)
        segCount += 1

sheet2 = wb.add_sheet('Sheet 2')
print(IS)
for i in range(0, len(IS)):
    sheet2.write(i, 0, str(obs[i, 0]))
    sheet2.write(i, 1, str(desOpts[i]))
    sheet2.write(i, 2, str(opts[i]))
    sheet2.write(i, 3, str(IS[i]))

edges = list(model.transitionGraph.edges)
for i in range(0, len(edges)):
    print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])
print("Options: ", model.nOptions)

wb.save('desired_option_check.xls')

print("Collection time: ", time.time() - stime)

