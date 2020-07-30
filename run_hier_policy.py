import os
import gym
import pybullet_envs
import time
import numpy as np
import hier_runner
import logger
import option_critic_model
from common import tf_util as U
from common.monitor import Monitor
import pickle

model_learning_params = {
    'per_train': 1,  # percentage of total rollouts to be trained
    'window_size': 2,  # window size of transition point clustering
    'weight_prior': 0.01,  # weight prior of DPGMM clustering for transition point
    'DBeps': 3.0,  # DBSCAN noise parameter for clustering segments
    'DBmin_samples': 2,  # DBSCAN minimum cluster size parameter for clustering segments
    'n_components': 2,  # number of DPGMM components to be used
    'minLength': 3,  # minimum segment length for Gaussian modelling
    'guassianEps': 1e-6,  # epsilon term added in Gaussian covariance
    'queueSize': 2500  # buffer size of samples
}
svm_grid_params = {
    'param_grid': {"C": np.logspace(-10, 10, endpoint=True, num=8, base=2.),
                   "gamma": np.logspace(-10, 10, endpoint=True, num=8, base=2.)},
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


def train(args, model_path=None, data_path=None):
    # Create TF session
    U.make_session().__enter__()

    def policy_fn(name, ob_space, ac_space, hybrid_model, num_options):
        return option_critic_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=[32, 32],
                                             model=hybrid_model, num_options=num_options, num_hid_layers=[2, 2],
                                             term_prob=0.5, eps=0.5)

    # Create environment
    env = gym.make(args.env)
    logger_path = logger.get_dir()
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(args.seed)

    # Train the policy using model based option actor critic
    pi, model = hier_runner.learn(env, model_path, data_path, policy_fn, model_learning_params, svm_grid_params,
                                  svm_params_interest, svm_params_guard, modes=args.modes, rolloutSize=args.rollouts,
                                  num_options=args.noptions,
                                  horizon=args.horizon,  # timesteps per actor per update
                                  clip_param=args.clip_param, ent_coeff=args.ent_coeff,
                                  # clipping parameter epsilon, entropy coeff
                                  optim_epochs=args.optim_epochs, optim_stepsize=args.optim_stepsize,
                                  optim_batchsize=args.optim_batchsize,  # optimization hypers
                                  gamma=args.gamma, lam=args.lam,  # advantage estimation
                                  max_iters=args.num_iteration,  # time constraint
                                  adam_epsilon=args.adam_epsilon,
                                  schedule='linear',  # annealing for stepsize parameters (epsilon and adam)
                                  retrain=args.retrain
                                  )
    if model_path:
        U.save_state(model_path+'/')
        model_file_name = model_path + '/hybrid_model.pkl'
        pickle.dump(model, open(model_file_name, "wb"), pickle.HIGHEST_PROTOCOL)
        print("Policy and Model saved in - ", model_path)
    return pi


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='BlockSlide2D-v1')
    parser.add_argument('--modes', help='Maximum modes expected in the environment', default=3, type=int)
    parser.add_argument('--noptions', help='Maximum options(edges) expected in the environment', default=9, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--horizon', help='Maximum time horizon in each episode', default=80, type=int)
    parser.add_argument('--rollouts', help='Maximum rollouts sampled in each iterations', default=100, type=int)
    parser.add_argument('--clip_param', help='Clipping parameter of PPO', default=0.25, type=float)
    parser.add_argument('--ent_coeff', help='Entropy coefficient of PPO', default=0.0, type=float)
    parser.add_argument('--optim_epochs', help='Maximum number of sub-epochs in optimization in each iteration', default=10, type=int)
    parser.add_argument('--optim_stepsize', help='Step size of sub-epochs in optimization in each iteration', default=3.25e-4, type=float)
    parser.add_argument('--optim_batchsize', help='Maximum number of samples in optimization in each iteration', default=100, type=int)
    parser.add_argument('--gamma', help='Discount factor of GAE', default=0.99, type=float)
    parser.add_argument('--lam', help='Lambda term of GAE', default=0.95, type=int)
    parser.add_argument('--adam_epsilon', help='Optimal step size', default=1e-4, type=float)
    parser.add_argument('--num_iteration', help='Number of training iteration', type=float, default=75)
    parser.add_argument('--retrain', help='Continued training, must provide saved model path', default=False, action='store_true')
    parser.add_argument('--exp_path', help='Path to logs,model and data', default=os.path.join(logger.get_dir(), 'block_ppo'), type=str)
    parser.add_argument('--play', help='Execute the trained policy', default=False, action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)
    log_path = args.exp_path + '/logs/'
    model_path = args.exp_path + '/model'
    data_path = args.exp_path + '/data'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    logger.configure(dir=log_path)

    if not args.play:
        # Train the Model
        train(args, model_path, data_path)
    else:
        # TODO fix replay and bullet rendering
        print("Setting up for replay")
        time.sleep(1)
        args.num_iteration = 1
        # Load the saved model for demonstration
        pi, model = train(args, model_path, data_path)
        U.load_state(model_path+'/')
        env = gym.make(args.env)
        env.setRender(True)
        ob = env.reset()
        option, active_options_t = pi.get_option(ob)
        time_step = 0
        model.currentMode = 0
        while True:
            ac = pi.act(True, ob, option)
            ob, rew, new, _ = env.step(ac)
            env.render()
            time_step = time_step + 1
            time.sleep(0.025)
            nbeta = pi.get_tpred(ob)
            tprob = nbeta[option]
            model.currentMode = model.getNextMode(ob)
            option, active_options_t = pi.get_option(ob)
            if tprob >= pi.term_prob:
                model.currentMode = model.getNextMode(ob)
                option, active_options_t = pi.get_option(ob)

            if new or time_step > args.horizon:
                ob = env.reset()
                time_step = 0
                option, active_options_t = pi.get_option(ob)


if __name__ == '__main__':
    main()
