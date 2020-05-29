from common.clustering_utils import hybridSegmentClustering
from common.model_learning_utils import learnTransitionRelation, multiDimGaussianProcess
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.svm import SVC
from copy import deepcopy
import time


# Hyperparameters for model learning
clustering_params = {
    'per_train': 1,  # percentage of total rollouts to be trained
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


trans_params = {
    'normalize': True,
    'constrain_ls': False,
    'ls_b_mul': (0.1, 10.),
    'constrain_sig_var': False,
    'sig_var_b_mul': (0.1, 10.),
    # 'noise_var': np.array([p_noise_var, v_noise_var]),
    'noise_var': None,
    'constrain_noise_var': False,
    'noise_var_b_mul': (1e-2, 1.),
    'fix_noise_var': True,
    'restarts': 1,
    }

expert_gpr_params = {
            'normalize': True,
            'constrain_ls': True,
            'ls_b_mul': (0.1, 100.),
            'constrain_sig_var': True,
            'sig_var_b_mul': (1e-1, 100.),
            # 'noise_var': np.array([p_noise_var, v_noise_var]),
            'noise_var': None,
            'constrain_noise_var': True,
            'noise_var_b_mul': (1e-1, 100.),
            'fix_noise_var': False,
            'restarts': 1,
        }

# Load log file
p = pickle.load(open("data/model_learning.pkl", "rb"))
epoch = 3
data = p[epoch]
rollouts = data['rollouts']
train_rollouts = rollouts[0:60]
test_rollouts = rollouts[60:]

#####################  Training hybrid model ############################

print("\n ..................... Learning hybrid dynamical model .....................\n")
start_time = time.time()
# Hybrid segmentation and clustering
nmodes, segmentedRollouts, x_train, u_train, delx_train, label_train, label_t_train = hybridSegmentClustering(test_rollouts, clustering_params)


print("Hybrid mode segmentation and clustering done\n")
label_train = np.squeeze(label_train)
label_t_train = np.squeeze(label_t_train)

# Transition relations and Return maps
if nmodes > 1:
    y_data, x_data = learnTransitionRelation(nmodes, segmentedRollouts)
    returnMap = multiDimGaussianProcess(trans_params)
    returnMap.fit(x_data, y_data)
    #Y_mu, Y_std = returnMap.predict(x_data)
    # rmse = mean_squared_error(y_data, Y_mu, squared=False)
    print("Transition Relation and Return Map learned\n")
else:
    transitionRel = []
    returnMap = []


# Guard functions
if nmodes > 1:
    guardFunction = SVC(**svm_params_guard)
    interestFunction = SVC(**svm_params_interest)
    interestFunction.fit(x_train, label_train)
    xu_train = np.hstack((x_train, u_train))
    guardFunction.fit(np.array(xu_train), label_t_train)
    print("Guard functions learned\n")
else:
    guardFunction = []
    interestFunction = []


# Transition dynamic model
modeDynamicsModel = []
for mode in range(0, nmodes):
    x_train_mode = x_train[(np.logical_and((label_train == mode), (label_t_train == mode)))]
    delx_train_mode = delx_train[(np.logical_and((label_train == mode), (label_t_train == mode)))]
    u_train_mode = u_train[(np.logical_and((label_train == mode), (label_t_train == mode)))]
    X = np.hstack((x_train_mode, u_train_mode))
    Y = delx_train_mode
    modeDynamics = multiDimGaussianProcess(expert_gpr_params)
    modeDynamics.fit(X, Y)
    modeDynamicsModel.append(modeDynamics)

print("Mode Dynamics learned\n")

#hybrid_model = {'dynamics': deepcopy(modeDynamicsModel), 'guardFunction': deepcopy(guardFunction), 'returnMap': deepcopy(returnMap)}
#pickle.dump(hybrid_model, open("hybridmodel.pkl","wb"))

## Sampling trajectory from learned model

rows = 4
cols = 5

for rollout_num in range(0, len(test_rollouts)):
    print("Rollout Num:", rollout_num)
    traj = test_rollouts[rollout_num]
    mode = 0
    data = []
    state = traj['observations'][0] #initial rollout
    mode = 0

    plt.subplot(rows, cols, rollout_num + 1)
    for t in range(0, len(traj['observations'])):
        action = traj['actions'][t]
        state_action_pair = np.hstack((state, action)).reshape(1, -1)
        # obtain next state
        delta_state_mu, delta_state_var = modeDynamicsModel[mode].predict(np.expand_dims(np.hstack((state, action)), axis=0))
        next_state = state + np.random.multivariate_normal(np.squeeze(delta_state_mu.T), np.diag(np.squeeze(delta_state_var.T))) #sampling from distribution
        if nmodes > 1:
            # obtain next mode
            next_mode = guardFunction.predict(state_action_pair) #deterministic
            if next_mode != mode:
                print("Mode Switched")
                next_state_mu, next_state_var = returnMap.predict(np.expand_dims(np.hstack((state, action, mode, next_mode)), axis=0))
                next_state = np.random.multivariate_normal(np.squeeze(next_state_mu.T), np.diag(np.squeeze(next_state_var.T)))
        else:
            next_mode = mode

        #data_dic = {'time': t, 'pobs': state, 'varobs': next_state_var, 'aobs': traj['observations'][t], 'action': action, 'cmode': mode, 'nmode': next_mode}
        actual_state = traj['observations'][t]
        if mode == 0:
            plt.plot(t, state[0], 'b*')
            plt.plot(t, actual_state[0], 'r*')
        else:
            plt.plot(t, state[0], 'k*')
            plt.plot(t, actual_state[0], 'g*')

        #data.append(data_dic)
        mode = int(next_mode)
        state = np.squeeze(next_state)

plt_name = "logs/model_learning_results_epoch"+str(epoch)+".png"
plt.savefig(plt_name)
plt.clf()

print("Total time: ", time.time()-start_time)
