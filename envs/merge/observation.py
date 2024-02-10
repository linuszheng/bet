from highway_env.envs.common.observation import ObservationType
from gym.spaces import Box, Discrete, Dict
import numpy as np
from .observation_helper import process_obs
from .settings import feature_indices, numHA, laneFinder, lanes_count, motor_model, lows, highs




class MyObservation(ObservationType):
  def __init__(self, env, base_obs):
    super().__init__(env)
    self.base_obs = base_obs
  def space(self):
      return Box(                   low=np.array(lows), 
                                  high=np.array(highs), 
                                  shape=(30,), dtype=np.float32)
  def observe(self):
    res = process_obs(self.env, self.base_obs.observe())
    # print(res[feature_indices])
    # print(self.get_lane())
    # print(f"crashed   {self.env.vehicle.crashed}   offroad   {not self.env.vehicle.on_road}")
    # res = res[feature_indices]
    res = np.append(res, self.env.last_action)
    res = np.append(res, motor_model(res))
    return res
  def get_lane(self):
    return laneFinder(process_obs(self.env, self.base_obs.observe())[1])
