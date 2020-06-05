import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

GOAL = np.array([0, 0.5])  # change here
INIT = np.array([-0.3, 0.8])  # pos1
# INIT = np.array([0.0, -0.1]) # pos2
# INIT = np.array([0.5, 0.3]) # pos3

ACTION_SCALE = 1e-3
# ACTION_SCALE = 1e-5
STATE_SCALE = 10


# TERMINAL_SCALE = 100
# T = 100
# EXP_SCALE = 2.


class Block2DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'block2D.xml', 1)
        self.reset_model()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = self.sim.data.qpos.flat[:2]
        dist = pos - GOAL
        reward_dist = -STATE_SCALE * np.linalg.norm(dist)
        reward_ctrl = -ACTION_SCALE * np.square(a).sum()
        reward = reward_dist + reward_ctrl
        done = False
        if np.linalg.norm(dist) < 0.0125:
            done = True
            reward = 100
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.distance = 4.0

    def reset_model(self):
        init_qpos = INIT
        init_qvel = np.zeros(2)
        self.set_state(init_qpos, init_qvel)
        # self.t = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            # self.model.data.qpos.flat[:7],
            # self.model.data.qvel.flat[:7],
            self.sim.data.qpos.flat[:2],
            self.sim.data.qvel.flat[:2],
            # self.get_body_com("blocky")[:2],
        ])
