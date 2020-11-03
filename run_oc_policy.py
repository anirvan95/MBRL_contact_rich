from common import tf_util as U
from common.monitor import Monitor
import logger
import oc_model
import oc_runner
import gym
import pybullet_envs
import os
import time


def train(args, model_path=None, data_path=None):
    # Create TF session
    U.make_session().__enter__()

    def policy_fn(name, ob_space, ac_space, num_options):
        return oc_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_options=num_options)

    # Create environment
    env = gym.make(args.env)
    logger_path = logger.get_dir()
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(args.seed)

    # Train the policy using PPO & GAE in actor critic fashion
    pi = oc_runner.learn(env, model_path, data_path, policy_fn, rolloutSize=args.rollouts,
                          num_options=args.noptions,
                          horizon=args.horizon,  # timesteps per actor per update
                          clip_param=args.clip_param, ent_coeff=args.ent_coeff,
                          # clipping parameter epsilon, entropy coeff
                          optim_epochs=args.optim_epochs, mainlr=args.main_lr,
                          intlr=args.int_lr, piolr=args.polo_lr, termlr=args.term_lr,
                          optim_batchsize=args.optim_batchsize,  # optimization hypers
                          gamma=args.gamma, lam=args.lam,  # advantage estimation
                          max_iters=args.num_iteration,  # time constraint
                          adam_epsilon=args.adam_epsilon,
                          schedule='linear',  # annealing for stepsize parameters (epsilon and adam)
                          retrain=args.retrain
                          )
    env.close()
    if model_path:
        U.save_state(model_path+'/')
        print("Policy Saved in - ", model_path)

    return pi


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='BlockInsert2D-v1')
    parser.add_argument('--noptions', help='Maximum options(edges) expected in the environment', default=3, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--horizon', help='Maximum time horizon in each episode', default=80, type=int)
    parser.add_argument('--rollouts', help='Maximum rollouts sampled in each iterations', default=100, type=int)
    parser.add_argument('--clip_param', help='Clipping parameter of PPO', default=0.2, type=float)
    parser.add_argument('--ent_coeff', help='Entropy coefficient of PPO', default=0.0, type=float)
    parser.add_argument('--optim_epochs', help='Maximum number of sub-epochs in optimization in each iteration',
                        default=10, type=int)
    parser.add_argument('--main_lr', help='Learning rate of intra option policy', default=1e-4, type=float)
    parser.add_argument('--term_lr', help='Learning rate of termination function', default=1e-4, type=float)
    parser.add_argument('--int_lr', help='Learning rate of interest function', default=1e-4, type=float)
    parser.add_argument('--polo_lr', help='Learning rate of option over policy', default=1e-4, type=float)
    parser.add_argument('--optim_batchsize', help='Maximum number of samples in optimization in each iteration',
                        default=100, type=int)
    parser.add_argument('--gamma', help='Discount factor of GAE', default=0.99, type=float)
    parser.add_argument('--lam', help='Lambda term of GAE', default=0.95, type=int)
    parser.add_argument('--adam_epsilon', help='Optimal step size', default=1e-4, type=float)
    parser.add_argument('--num_iteration', help='Number of training iteration', type=float, default=50)
    parser.add_argument('--retrain', help='Continued training, must provide saved model path', default=False,
                        action='store_true')
    parser.add_argument('--exp_path', help='Path to logs,model and data',
                        default=os.path.join(logger.get_dir(), 'block_ppo'), type=str)
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
    print(args.play)
    if not args.play:
        # Train the Model
        train(args, model_path, data_path)
    else:
        print("Setting up for replay")
        rollout = 0
        time.sleep(1)
        args.num_iteration = 1
        # Load the saved model for demonstration
        pi = train(args, model_path, data_path)
        U.load_state(model_path+'/')
        env = gym.make(args.env)
        env.setRender(True)
        print("Setting up for replay")
        time.sleep(1)
        while rollout < 10:
            print("Rollout : ", rollout)
            ob = env.reset()
            option, active_options_t = pi.get_option(ob)
            for i in range(0, args.horizon):
                ac = pi.act(False, ob, option)
                print(ac)
                ob, reward, done, _ = env.step(ac)
                term = pi.get_term([ob], [option])[0][0]
                if term:
                    option = pi.get_option(ob)
                env.render()
                time.sleep(0.01)

            rollout = rollout + 1
        print("Done")
        time.sleep(5)


if __name__ == '__main__':
    main()
