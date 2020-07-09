from model_learning import partialHybridModel
import pickle
import numpy as np

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


f = open("data/base_actor_critic_blcSlid1.pkl", "rb")
p = pickle.load(f)
f.close()
epoch_number = 22
data = p[epoch_number]
rollouts = data['seg']
horizon = 150
rolloutSize = 50
modes = 4
options = 12
model = partialHybridModel(clustering_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, options, rolloutSize)
model.learnHardTG(rollouts)
print("Done")
print(list(model.transitionGraph.nodes))
print(list(model.transitionGraph.edges))
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

