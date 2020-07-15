from model_learning import partialHybridModel
import pickle
import numpy as np
import time

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


f = open("data/exp_blcIn_1.pkl", "rb")
p = pickle.load(f)
f.close()
horizon = 150
rolloutSize = 20
modes = 2
options = 4
queueSize = 5000
model = partialHybridModel(clustering_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, options, rolloutSize, queueSize)
i = 7
data = p[i]
rollouts = data['seg']
stime = time.time()
model.updateModel(rollouts)
model.currentMode = 1
mode1Data = model.dataset[1]
x1 = mode1Data[0]['x']
print(model.getNextMode(x1))

'''
print(model.transitionGraph.nodes)
mode0Data = model.dataset[0]
x0 = mode0Data[0]['x']
print(model.getInterest(x0))


model.learnGuardF()
model.learnModeF()
mode0Data = model.dataset[0]
x0 = mode0Data[0]['x']
mode1Data = model.dataset[1]
x1 = mode1Data[0]['x']
mode2Data = model.dataset[2]
x2 = mode2Data[1]['x']
print(model.modeFunction.predict([x1]))
print(model.guardFunction.predict_f([x1]))
model.currentMode = 0
print("Interest Check !!")
print(model.getInterest(x0))
model.currentMode = 1
print(model.getInterest(x1))
print("Termination Check !!")
model.currentMode = 0
print(model.getTermination(x0))
model.currentMode = 1
print(model.getTermination(x1))
model.currentMode = 2
print(model.getTermination(x2))
print("Testing model learning")
model.learnGuardF()
model.learnModeF()
'''
