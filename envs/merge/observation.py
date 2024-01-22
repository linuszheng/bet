from highway_env.envs.common.observation import ObservationType
from gym.spaces import Box, Discrete, Dict
import numpy as np
from .observation_helper import process_obs
from .settings import feature_indices, numHA, laneFinder, lanes_count




class MyObservation(ObservationType):
  def __init__(self, env, base_obs):
    super().__init__(env)
    self.base_obs = base_obs
  def space(self):
      return Box(                   low=np.array([0, 0, 0, 0, 0]), 
                                  high=np.array([600, 60, 600, 600, 600]), 
                                  shape=(5,), dtype=np.float32)
  def observe(self):
    res = process_obs(self.env, self.base_obs.observe())
    # print(res[feature_indices])
    # print(self.get_lane())
    # print(f"crashed   {self.env.vehicle.crashed}   offroad   {not self.env.vehicle.on_road}")
    return res[feature_indices]
  def get_lane(self):
    return laneFinder(process_obs(self.env, self.base_obs.observe())[1])
