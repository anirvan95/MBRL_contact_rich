from common import tf_util as U
from common.monitor import Monitor
import logger
import actor_critic_model
import bac_runner
import gym
import pybullet_envs
import os
import time


def train(args, model_path=None, data_path=None):
    # Create TF session
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return actor_critic_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2)

    # Create environment
    env = gym.make(args.env)
    logger_path = logger.get_dir()
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(args.seed)

    # Train the policy using PPO & GAE in actor critic fashion
    pi = bac_runner.learn(env, model_path, data_path, policy_fn, horizon=args.horizon, rolloutSize=args.rollouts,
                          clip_param=args.clip_param, entcoeff=args.ent_coeff,
                          optim_epochs=args.optim_epochs, optim_stepsize=args.optim_step_size, optim_batchsize=args.optim_batchsize,
                          gamma=args.gamma, lam=args.lam, max_iters=args.num_iteration,
                          adam_epsilon=args.adam_epsilon, schedule='linear',
                          retrain=args.retrain
                          )
    if model_path:
        U.save_state(model_path)
        print("Policy Saved in - ", model_path)

    return pi


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='BlockInsert2D-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--horizon', help='Maximum time horizon in each episode', default=80, type=int)
    parser.add_argument('--rollouts', help='Maximum rollouts sampled in each iterations', default=100, type=int)
    parser.add_argument('--clip_param', help='Clipping parameter of PPO', default=0.2, type=float)
    parser.add_argument('--ent_coeff', help='Entropy coefficient of PPO', default=0.0, type=float)
    parser.add_argument('--optim_epochs', help='Maximum number of sub-epochs in optimization in each iteration', default=15, type=int)
    parser.add_argument('--optim_step_size', help='Step size of sub-epochs in optimization in each iteration', default=3e-4, type=float)
    parser.add_argument('--optim_batchsize', help='Maximum number of samples in optimization in each iteration', default=100, type=int)
    parser.add_argument('--gamma', help='Discount factor of GAE', default=0.99, type=float)
    parser.add_argument('--lam', help='Lambda term of GAE', default=0.95, type=int)
    parser.add_argument('--adam_epsilon', help='Optimal step size', default=1e-4, type=float)
    parser.add_argument('--num_iteration', help='Number of training iteration', type=float, default=50)
    parser.add_argument('--retrain', help='Continued training, must provide saved model path', default=False, action='store_true')
    parser.add_argument('--exp_path', help='Path to logs,model and data', default=os.path.join(logger.get_dir(), 'BAC_exp'), type=str)
    parser.add_argument('--play', help='Execute the trained policy', default=False, action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)

    log_path = args.exp_path + '/logs/'
    model_path = args.exp_path + '/model/'
    data_path = args.exp_path + '/data/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    logger.configure(dir=log_path)

    if not args.play:
        # Train the Model
        print("Training started.. !!")
        train(args, model_path, data_path)
    else:
        print("Setting up for replay")
        rollout = 0
        time.sleep(1)
        args.num_iteration = 1
        # Load the saved model for demonstration
        pi = train(args, model_path, data_path)
        U.load_state(model_path)
        env = gym.make(args.env)
        env.setRender(True)
        ob = env.reset()
        time.sleep(60)
        while rollout < 10:
            print("Rollout : ", rollout)
            for i in range(0, args.horizon):
                action = pi.act(stochastic=False, ob=ob)[0]
                ob, reward, done, _ = env.step(action)
                env.render()
                time.sleep(0.01)

            ob = env.reset()
            rollout = rollout + 1
        time.sleep(5)


if __name__ == '__main__':
    main()
