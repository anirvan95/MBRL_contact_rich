"""
2D Block Sliding envirionment
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

'''
# Change environment configuration here
'''
GOAL = np.array([0.15, 0.15])
INIT = np.array([0.45, 0.55, 0.15])

ACTION_SCALE = 2e-2
STATE_SCALE = 3
EXPONENT_SCALE = 10
gForce = -4.9


class BlockSlide2DEnvC1(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, render=False):
        self._renders = render
        self._render_height = 200
        self._render_width = 320
        self._physics_client_id = -1
        actuator_bound_low = np.array([-10, -10])
        actuator_bound_high = np.array([10, 10])
        self.action_space = spaces.Box(low=actuator_bound_low, high=actuator_bound_high)
        observation_dim = 4
        state_bound_low = np.full(observation_dim, -float('inf'))
        state_bound_high = np.full(observation_dim, float('inf'))
        self.observation_space = spaces.Box(low=state_bound_low, high=state_bound_high)
        self.seed()
        self.initConnection = True
        self.viewer = None
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def getGoalDist(self):
        dist = self.state[0:2] - GOAL
        return dist

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        p = self._p
        forceX = action[0]
        forceY = action[1]
        p.setJointMotorControl2(self.block, 0, p.TORQUE_CONTROL, force=forceX)
        p.setJointMotorControl2(self.block, 1, p.TORQUE_CONTROL, force=forceY)
        p.setJointMotorControl2(self.block, 2, p.TORQUE_CONTROL, force=-4.9)
        p.stepSimulation()
        self.state = [p.getJointState(self.block, 0)[0], p.getJointState(self.block, 1)[0], p.getJointState(self.block, 0)[1], p.getJointState(self.block, 1)[1]]
        done = False
        pos = self.state[0:2]
        dist = pos - GOAL
        reward_dist = -STATE_SCALE * np.linalg.norm(dist)
        reward_ctrl = -ACTION_SCALE * np.square(action).sum()
        reward = EXPONENT_SCALE*np.exp(reward_dist + reward_ctrl)

        return np.array(self.state), reward, done, {}

    def setRender(self, render):
        self._renders = render
        self.initConnection = True

    def reset(self):
        if self.initConnection:
            self.initConnection = False
            if self._renders:
                self._p = bc.BulletClient(connection_mode=p2.GUI)
            else:
                self._p = bc.BulletClient()
            self._physics_client_id = self._p._client

            p = self._p
            p.resetSimulation()
            p.resetDebugVisualizerCamera(cameraDistance=3.78, cameraYaw=134.0, cameraPitch=-45.8, cameraTargetPosition=[0.03, -0.04, 0.03])
            self.wall_1 = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block_slide_w1.urdf"))
            p.changeDynamics(self.wall_1, 0, lateralFriction=0.01)
            p.changeDynamics(self.wall_1, 1, lateralFriction=0.01)
            self.wall_2 = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block_slide_w2.urdf"))
            p.changeDynamics(self.wall_2, 0, lateralFriction=0.01)
            p.changeDynamics(self.wall_2, 1, lateralFriction=0.01)
            self.floor = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block_slide_wm3.urdf"))
            p.changeDynamics(self.floor, 0, lateralFriction=0.01)
            p.changeDynamics(self.floor, 1, lateralFriction=0.01)
            p.changeDynamics(self.floor, 2, lateralFriction=0.01)
            p.changeDynamics(self.floor, 3, lateralFriction=0.8)
            #print(p.getDynamicsInfo(self.floor, 3))
            self.block = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "block3D.urdf"))
            p.changeDynamics(self.block, 0, lateralFriction=0.55)
            p.changeDynamics(self.block, 1, lateralFriction=0.55)
            p.changeDynamics(self.block, 2, lateralFriction=0.55)
            self.timeStep = 0.025
            self.mode = 0
            p.setGravity(0, 0, 0)
            p.setTimeStep(self.timeStep)
            p.setRealTimeSimulation(0)
            p.setJointMotorControl2(self.block, 0, p.VELOCITY_CONTROL, force=0)
            p.setJointMotorControl2(self.block, 1, p.VELOCITY_CONTROL, force=0)
            p.setJointMotorControl2(self.block, 2, p.VELOCITY_CONTROL, force=0)

        p = self._p
        p.resetJointState(self.block, 0, INIT[0], 0)
        p.resetJointState(self.block, 1, INIT[1], 0)
        p.resetJointState(self.block, 2, INIT[2], 0)
        self.state = [p.getJointState(self.block, 0)[0], p.getJointState(self.block, 1)[0], p.getJointState(self.block, 0)[1], p.getJointState(self.block, 1)[1]]

        return np.array(self.state)

    def render(self, mode='human', close=False):
        base_pos = [0, 0, 0]
        self._cam_dist = 2
        self._cam_pitch = 0.3
        self._cam_yaw = 0

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
        px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        self._p.disconnect()
        self._physics_client_id = -1

    def getContactForce(self):
        contactInfo = self._p.getContactPoints(self.block)
        #print(contactInfo)
        dirX = np.array([1, 0, 0])
        dirY = np.array([0, 1, 0])
        dirZ = np.array([0, 0, 1])
        cForceX = []
        cForceY = []
        cForceZ = []
        nForceX = []
        nForceY = []
        nForceZ = []

        nFx = 0.0
        nFy = 0.0
        nFz = 0.0
        cFx = 0.0
        cFy = 0.0
        cFz = 0.0
        if len(contactInfo) > 0:
            for i in range(0, len(contactInfo)):
                normalDir = np.array(contactInfo[i][7])
                normalForce = float(contactInfo[i][9])
                nForceX.append(np.multiply(normalDir, dirX) * normalForce)
                nForceY.append(np.multiply(normalDir, dirY) * normalForce)
                nForceZ.append(np.multiply(normalDir, dirZ) * normalForce)
                fricDir1 = np.array(contactInfo[i][11])
                fricForce1 = float(contactInfo[i][10])
                fricDir2 = np.array(contactInfo[i][13])
                fricForce2 = float(contactInfo[i][12])
                cForceX.append(np.multiply(fricDir1, dirX) * fricForce1 + np.multiply(fricDir2, dirX) * fricForce2)
                cForceY.append(np.multiply(fricDir1, dirY) * fricForce1 + np.multiply(fricDir2, dirY) * fricForce2)
                cForceZ.append(np.multiply(fricDir1, dirZ) * fricForce1 + np.multiply(fricDir2, dirZ) * fricForce2)

            nFx = np.mean(np.array(nForceX))
            nFy = np.mean(np.array(nForceY))
            nFz = np.mean(np.array(nForceZ))
            cFx = np.mean(np.array(cForceX))
            cFy = np.mean(np.array(cForceY))
            cFz = np.mean(np.array(cForceZ))
            nFx = 0.0 if math.isnan(nFx) else nFx
            nFy = 0.0 if math.isnan(nFy) else nFy
            nFz = 0.0 if math.isnan(nFz) else nFz
            cFx = 0.0 if math.isnan(cFx) else cFx
            cFy = 0.0 if math.isnan(cFy) else cFy
            cFz = 0.0 if math.isnan(cFz) else cFz

        return np.array([nFx, nFy, nFz, cFx, cFy, cFz])
