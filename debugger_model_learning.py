from model_learning import partialHybridModel
import pickle
import numpy as np
import time
import gym
import pybullet_envs
import xlwt
from xlwt import Workbook

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
from common.model_learning_utils import *

f = open("results/MOAC/exp_5/data/rollout_data.pkl", "rb")
p = pickle.load(f)
f.close()
horizon = 150
rolloutSize = 75
modes = 3
num_options = 9
queueSize = 5000
env = gym.make('BlockSlide2D-v1')
model = partialHybridModel(env, model_learning_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, num_options, rolloutSize)
i = 35
data = p[i]
rollouts = data['seg']
env_id = env.unwrapped.spec.id
train_rollouts = int(model_learning_params['per_train']) * rolloutSize
obs = rollouts['ob']
opts = rollouts['opts']
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
for rollout in range(0, train_rollouts-1):
    print("Rollout: ", rollout)
    opt = opts[rollout * horizon:(horizon + rollout * horizon)]
    rolloutIndex = rollout * horizon
    numSegments = len(segment_data[rollout][1])
    for segment in range(0, numSegments):
        print("Segment Num: ", segment)
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

                    print(list(model.transitionGraph.edges))
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

        xl_sheet_string = 'seg : ' + str(segment_data[rollout][1][segment]) + ', l :' + str(labels[segCount]) + ', do :' + str(desOptionsLabels[segCount])
        sheet.write(rollout, segment, xl_sheet_string)
        segCount += 1

sheet = wb.add_sheet('Sheet 2')
edges = list(model.transitionGraph.edges)
for i in range(0, len(edges)):
    print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])
print("Options: ", model.nOptions)

wb.save('desired_option_check.xls')