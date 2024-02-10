import numpy as np
import gym
from gym import spaces
from .settings import pv_stddev, pv_range, motor_model
from panda_gym.envs import PandaPickAndPlaceEnv
from gym.utils import seeding

EPSILON = 10E-10
n_timesteps = 50


obs_min = [-0.14771309999999999, -0.23160631250000002, 0.0048890291249999995, -0.1527093125, -0.24460781250000002, 0.024538135000000003, -0.11967610625, -0.160605425, 0.025, 4.29890925e-08, -1.2241278620738945, -1.1626909868023398, -1.25, -1.25, -1.002689975, -0.908584359, -1.1088294125, 0.75, -1.3817946, -1.01701121875, -1.1088294125, -0.75]
obs_max = [0.17073798750000002, 0.1744322625, 0.2467495375, 0.174665675, 0.2001017, 0.1890869125, 0.1738784625, 0.175174, 0.1727453, 0.100059075, 1.097475279687433, 1.25, 0.8956591423000597, 1.25, 0.6145945625, 0.5118701035, 0.10485015624999999, 0.75, 0.6145945625, 1.586372625, 0.8245543962500002, 0.375]


class PickPlaceEnv(gym.Env):

  panda_env = PandaPickAndPlaceEnv()

  def __init__(self):
    self.observation_space = spaces.Box(   low=np.array(obs_min), 
                                  high=np.array(obs_max), 
                                  shape=(22,), dtype=np.float32)
    self.action_space = spaces.Box(shape=(4,), low=np.array([-2,-2,-2,-2]), high=np.array([2,2,2,2]), dtype=np.float32)
    self.last_run_was_success = False
    self.n_success = 0
    self.reset()
    
  
  def reset(self):
    if self.last_run_was_success:
      self.last_run_was_success = False
      self.n_success += 1
    print(self.n_success)
    self.panda_env.reset()
    self.t = 0
    self.last_la = [0,0,0,0]
    return self._get_obs()

  def _get_obs(self):
    observation = self.panda_env._get_obs()
    world_state = observation["observation"]
    target_pos = observation["desired_goal"][0:3]
    x, y, z, bx, by, bz, tx, ty, tz, end_width = world_state[0], world_state[1], world_state[2], world_state[7], world_state[8], world_state[9], target_pos[0], target_pos[1], target_pos[2], world_state[6]
    obs_pruned = [x, y, z, bx, by, bz, tx, ty, tz, end_width]
    obs_pruned = np.concatenate((obs_pruned, self.last_la))
    return np.concatenate((obs_pruned, motor_model(obs_pruned)))

  def _get_info(self):
    return {}

  def step(self, action):
    self.t += 1
    self.last_run_was_success = self.panda_env.step(action)[2]
    self.last_la = action
    return self._get_obs(), 0, self.t > n_timesteps, {}
  

  def render(self, mode="human"):
    pass



gym.register(
    id="pickplace-v0",
    entry_point=PickPlaceEnv
)

