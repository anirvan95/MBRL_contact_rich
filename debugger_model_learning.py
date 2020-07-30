from model_learning import partialHybridModel
import pickle
import numpy as np
import time
import gym
import pybullet_envs
import xlwt
from xlwt import Workbook
import matplotlib.pyplot as plt
import option_critic_model
import common.tf_util as U
wb = Workbook()
sheet = wb.add_sheet('Sheet 1')

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
from common.model_learning_utils import *
def policy_fn(name, ob_space, ac_space, hybrid_model, num_options):
    return option_critic_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=[35, 25],
                                         model=hybrid_model, num_options=num_options, num_hid_layers=[2, 2],
                                         term_prob=0.5, eps=0.2)


f = open("results/MOAC/isexp2_5/data/rollout_data.pkl", "rb")
p = pickle.load(f)
f.close()
horizon = 200
rolloutSize = 60
modes = 3
num_options = 9
queueSize = 2500
env = gym.make('BlockSlide2D-v1')
ob_space = env.observation_space
ac_space = env.action_space
model = partialHybridModel(env, model_learning_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, num_options, rolloutSize)
pi = policy_fn("pi", ob_space, ac_space, model, num_options)  # Construct network for new policy

U.initialize()



i = 1
data = p[i]
rollouts = data['rollouts']
model.learnPreDefModes(rollouts)
model.learnTranstionRelation(rollouts, pi)
print(model.des_opts_s)
'''
env_id = env.unwrapped.spec.id
train_rollouts = int(model_learning_params['per_train']) * rolloutSize
obs = rollouts['ob']
opts = rollouts['opts']
desOpts = opts
segmented_traj = []
seg_labels = []
fig = plt.figure()
wb = Workbook()
sheet = wb.add_sheet('Sheet 1')

for rollout in range(0, rolloutSize):
    states = obs[rollout * horizon:(horizon + rollout * horizon), :]
    traj_time = len(states)
    tp = []
    ax = fig.add_subplot(6, 10, rollout+1)
    for t in range(0, traj_time - 1):
        label = obtainMode(env_id, states[t, :])
        label_t = obtainMode(env_id, states[t + 1, :])
        dataDict = {'x': states[t, :], 'label': label, 'label_t': label_t}
        if label == 0:
            ax.plot(states[t, 0], states[t, 1], '.r-')
        elif label == 1:
            ax.plot(states[t, 0], states[t, 1], '.b-')
        elif label == 2:
            ax.plot(states[t, 0], states[t, 1], '.g-')
        model.dataset[int(label)].append(dataDict)
        if label != label_t:
            tp.append(t)
    tp.append(0)
    tp.append(traj_time-1)
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
print(labels)
print(segment_data)
plt.show()
plt.savefig('tester.png')

seg_count = 0
des_opts_seg = np.zeros(labels.shape)
des_opts = []
imp_samp = []
des_act = []

seg_obs = []
seg_acs = []
seg_opts = []
seg_rews = []
seg_news = []
print("Rejected options", model.rejectedOptions)
for rollout in range(0, rolloutSize-1):
    option = opts[rollout * model.horizon:(model.horizon + rollout * model.horizon)]
    states = obs[rollout * model.horizon:(model.horizon + rollout * model.horizon), :]
    num_segments = len(segment_data[rollout][1])
    print("Rollout ", rollout, "seg :", num_segments)
    for segment in range(0, num_segments):
        print("Segment num = ", segment, "label: ", labels[seg_count])
        # Avoid Noisy segments
        if labels[seg_count] >= 0:
            # Adding mode to GOAL edge e.g. 0 > goal, 1 > goal etc
            if not model.transitionGraph.has_edge(labels[seg_count], 'goal'):
                model.nOptions += 1
                model.transitionGraph.add_weighted_edges_from([(model.labels[seg_count], 'goal', model.nOptions)])
                model.transitionUpdated = False

            if labels[seg_count + 1] >= 0:  # Avoid noisy next segment
                # Adding mode to GOAL edge e.g. 1 > goal, 1 > goal etc
                if not model.transitionGraph.has_edge(labels[seg_count + 1], 'goal'):
                    model.nOptions += 1
                    model.transitionGraph.add_weighted_edges_from([(labels[seg_count + 1], 'goal', model.nOptions)])

                # Checking for transition detection while ignoring the last segment
                if labels[seg_count] != labels[seg_count + 1] and segment < num_segments - 1:
                    print("Assigning Transition and desired option ", labels[seg_count+1])
                    if not (model.transitionGraph.has_edge(labels[seg_count], labels[seg_count + 1])):
                        model.nOptions += 1
                        model.transitionGraph.add_weighted_edges_from([(labels[seg_count], labels[seg_count + 1], model.nOptions)])

                    # Assign the desired option for transition
                    # 0>1
                    if not [labels[seg_count], labels[seg_count+1]] in model.rejectedOptions:

                        des_opts_seg[seg_count] = model.transitionGraph[labels[seg_count]][labels[seg_count + 1]]['weight']
                        print("Assigned, ", des_opts_seg[seg_count])
                    # 1>0 gets assigned to 1>goal
                    else:
                        des_opts_seg[seg_count] = model.transitionGraph[labels[seg_count]]['goal']['weight']
                        print("Option in rejected list, Assigned, ", des_opts_seg[seg_count])
                # Assigning desired option for last segment or only one segment
                else:
                    print("No transition found, ", des_opts_seg[seg_count])
                    des_opts_seg[seg_count] = model.transitionGraph[labels[seg_count]]['goal']['weight']

            # Creating the updated database
            segStates = states[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1]+1), :]
            segOpts = option[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1]+1)]
            print("Segment states length, ", len(segStates))
            if seg_count == 0:
                seg_obs = segStates
                seg_opts = segOpts
            else:
                seg_obs = np.vstack((seg_obs, segStates))
                seg_opts = np.append(seg_opts, segOpts)
            # Compute importance sampling ration
            is_ratio = 1
            for t in range(0, len(segStates)):
                des_opts.append(des_opts_seg[seg_count])
        xl_sheet_string = 'seg : ' + str(segment_data[rollout][1][segment]) + ', l :' + str(labels[seg_count]) + ', do :' + str(des_opts_seg[seg_count])
        sheet.write(rollout, segment, xl_sheet_string)
        seg_count += 1
print("Model graph:", model.transitionGraph.nodes)
print("Model options:", model.transitionGraph.edges)
edges = list(model.transitionGraph.edges)
for i in range(0, len(edges)):
    print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])
print(des_opts_seg)

wb.save('desired_option_check.xls')
'''