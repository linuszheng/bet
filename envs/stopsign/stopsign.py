import numpy as np
import gym
from gym import spaces
from .settings import numHA, n_timesteps, pv_stddev, motor_model


EPSILON = 10E-10

class StopSignEnv(gym.Env):

    def __init__(self):
      self.dt = .1
      self.pos = 0.
      self.vel = 0.
      self.acc = 0.
      self.prev_acc = 0.
      self.prev_ha = 0
      self.decMax = -5.
      self.accMax = 6.
      self.vMax = 10.
      self.target = 100.
      self.observation_space = spaces.Box( low=np.array([0, -40, 0, 0, 0, 100] + [-40]*4), 
                                  high=np.array([150, 0, 40, 250, 250, 100] + [40]*4), 
                                  shape=(10,), dtype=np.float32)
      self.action_space = spaces.Box(shape=(1,), low=np.array([-40]), high=np.array([40]), dtype=np.float32)
      self.t = 0
      self.action_list = []

    def config(self, decMax, accMax, vMax, target):
      self.decMax = float(decMax)
      self.accMax = float(accMax)
      self.vMax = float(vMax)
      self.target = float(target)

    def _get_info(self):
      return { }

    def _get_obs(self):
      lim_obs = [self.pos, self.decMax, self.accMax, self.vMax, self.vel, self.target, self.acc]
      lim_obs = lim_obs + motor_model(lim_obs)
      return np.array(lim_obs, dtype=np.float32)


    def reset(self, seed=None, options=None):
      # super().reset(seed=seed)
      self.pos = 0.
      self.vel = 0.
      self.acc = 0.
      self.t = 0.
      self.prev_ha = 0
      return self._get_obs()

    def step(self, action):
      prev_vel = self.vel
      self.acc = action
      self.vel = self.vel+self.acc*self.dt
      if self.vel < EPSILON:
        self.vel = 0
      if abs(self.vel - self.vMax) < EPSILON:
        self.vel = self.vMax
      if abs(self.pos - self.target) < EPSILON:
        self.pos = self.target
      self.pos += (prev_vel + self.vel)*.5*self.dt
      if self.t == 0:
        self.acc = 1.
      self.t += 1
      self.action_list.append(self.acc)
      print(self.acc)
      return self._get_obs(), 0, self.t > n_timesteps, self._get_info()

    def render(self, mode="human"):
      print(self._get_obs())
    
    def close(self):
      # print(self.action_list)
      pass
    





gym.register(
    id="stopsign-v0",
    entry_point=StopSignEnv
)

