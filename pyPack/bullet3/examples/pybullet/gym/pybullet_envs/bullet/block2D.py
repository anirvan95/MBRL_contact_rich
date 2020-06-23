"""
2D Block Insertion environment converted from Mujoco
"""
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p2
import pybullet_data
import pybullet_utils.bullet_client as bc
from pkg_resources import parse_version

logger = logging.getLogger(__name__)

GOAL = np.array([0, 0.52])  # change here
INIT = np.array([-0.3, 0.8])  # pos1
# INIT = np.array([0.0, -0.1]) # pos2
# INIT = np.array([0.5, 0.3]) # pos3

ACTION_SCALE = 1e-3
# ACTION_SCALE = 1e-5
STATE_SCALE = 10


class Block2DEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, render=False):
        self._renders = render
        self._render_height = 200
        self._render_width = 320
        self._physics_client_id = -1
        actuator_bound_low = np.array([-10, -10])
        actuator_bound_high = np.array([10, 10])
        self.action_space =  spaces.Box(low=actuator_bound_low, high=actuator_bound_high)
        observation_dim = 4
        state_bound_low = np.full(observation_dim, -float('inf'))
        state_bound_high = np.full(observation_dim, float('inf'))
        self.observation_space = spaces.Box(low=state_bound_low, high=state_bound_high)
        self.seed()
        self.viewer = None
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        p = self._p
        forceX = action[0]
        forceY = action[1]
        p.setJointMotorControl2(self.block, 0, p.TORQUE_CONTROL, force=forceX)
        p.setJointMotorControl2(self.block, 1, p.TORQUE_CONTROL, force=forceY)
        p.stepSimulation()
        self.state = [p.getJointState(self.block, 0)[0], p.getJointState(self.block, 1)[0], p.getJointState(self.block, 0)[1], p.getJointState(self.block, 1)[1]]
        done = False
        pos = self.state[0:2]
        dist = pos - GOAL
        reward_dist = -STATE_SCALE * np.linalg.norm(dist)
        reward_ctrl = -ACTION_SCALE * np.square(action).sum()
        reward = reward_dist + reward_ctrl

        return np.array(self.state), reward, done, {}

    def reset(self):
        #    print("-----------reset simulation---------------")

        if self._renders:
            self._p = bc.BulletClient(connection_mode=p2.GUI)
        else:
            self._p = bc.BulletClient()
        self._physics_client_id = self._p._client

        p = self._p
        p.resetSimulation()
        self.slot_1 = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block_insert_fl1.urdf"))
        self.slot_2 = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block_insert_fl2.urdf"))
        self.slot_3 = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block_insert_fl3.urdf"))
        self.block = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block.urdf"))
        self.timeStep = 0.01
        p.setGravity(0, 0, 0)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)
        p.setJointMotorControl2(self.block, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.block, 1, p.VELOCITY_CONTROL, force=0)

        #p = self._p
        p.resetJointState(self.block, 0, INIT[0], 0)
        p.resetJointState(self.block, 1, INIT[1], 0)
        self.state = [p.getJointState(self.block, 0)[0], p.getJointState(self.block, 1)[0], p.getJointState(self.block, 0)[1], p.getJointState(self.block, 1)[1]]

        return np.array(self.state)

    def render(self, mode='human', close=False):
        if mode == "human":
            self._renders = True
        if mode != "rgb_array":
            return np.array([])
        base_pos = [0, 0, 0]
        self._cam_dist = 2
        self._cam_pitch = 0.3
        self._cam_yaw = 0
        if (self._physics_client_id >= 0):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(self._render_width) /
                                                                    self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
        else:
            px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        self._renders = False
        if self._physics_client_id >= 0:
            self._p.disconnect()
        self._physics_client_id = -1

    def getContactForce(self):
        contactPoint = self._p.getContactPoints(self.block)
        sumContactForce = 0
        contactForce = 0
        count = 0
        validContact = False
        if len(contactPoint) > 0:
            for i in range(0, len(contactPoint)):
                if contactPoint[i][9] > 0:
                    sumContactForce = sumContactForce + contactPoint[i][9]
                    count += 1
                    validContact = True
            if validContact:
                contactForce = sumContactForce / count
            else:
                contactForce = 0
        return contactForce




