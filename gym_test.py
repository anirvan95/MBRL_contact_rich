import gym
import pybullet_envs
import time

env = gym.make('BlockSlide2Dc-v2')
env.setRender(True)
env.reset()
time.sleep(60)