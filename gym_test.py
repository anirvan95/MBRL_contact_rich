import gym
import pybullet_envs
import time

env = gym.make('BlockInsert2Dc-v1')
for iteration in range(0, 5):
    print("Iteration, ", iteration)
    render = True
    for rollouts in range(0, 10):
        if render:
            env.setRender(True)
        else:
            env.setRender(False)
        env.reset()
        for _ in range(80):
            a = env.action_space.sample()
            print(a)
            env.step(a)  # take a random action
            env.render()
            #time.sleep(0.1)
        render = False