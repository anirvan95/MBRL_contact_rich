from common import tf_util as U
from common.monitor import Monitor
import logger
import ioc_model, ioc_runner
import gym
import os


def train(env_id, num_iteration, seed, log_path=None, model_path=None):
    # Create TF session
    U.make_session().__enter__()

    def policy_fn(name, ob_space, ac_space):
        return ioc_model.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_options=2, dc=0, w_intfc=True, k=0)

    # Create Mujoco environment
    env = gym.make(env_id)
    logger_path = logger.get_dir()
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)

    # Train the policy using PPO & GAE in actor critic fashion
    # Tune hyperparameters here, will be moved to main args for grid search
    pi = ioc_runner.learn(env, model_path, policy_fn, num_options=2,
                          horizon=150, batchSize=int(150 * 80),  # timesteps per actor per update
                          clip_param=0.2, entcoeff=0.02,
                          optim_epochs=30, optim_stepsize=2.5e-4, optim_batchsize=32,  # optimization hypers
                          gamma=0.99, lam=0.95,  # advantage estimation
                          max_timesteps=3e6, max_episodes=0, max_iters=num_iteration, max_seconds=0,  # time constraint
                          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
                          adam_epsilon=1e-4,
                          schedule='linear',  # annealing for stepsize parameters (epsilon and adam),
                          retrain=True
                          )
    env.close()
    if model_path:
        U.save_state(log_path)
        print("Model saved")

    return pi


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='Block2D-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--num_iteration', type=float, default=50)
    parser.add_argument('--model_path', help='Path to save trained model to',
                        default=os.path.join(logger.get_dir(), 'block_ppo'), type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--horizon', help='Maximum time horizon in each iteration', default=50, type=int)

    args = parser.parse_args()
    logger.configure(dir=args.log_path)

    if not args.play:
        # Train the Model
        train(args.env, num_iteration=args.num_iteration, seed=args.seed, log_path=args.log_path, model_path=args.model_path)
    else:
        # Load the saved model for demonstration
        pi = train(args.env, num_iteration=1, seed=args.seed)
        U.load_state(args.model_path)
        env = gym.make(args.env)
        ob = env.reset()
        time_step = 0
        while True:

            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, = env.step(action)
            env.render()
            time_step = time_step + 1
            if done or time_step > args.horizon:
                ob = env.reset()
                time_step = 0


if __name__ == '__main__':
    main()
