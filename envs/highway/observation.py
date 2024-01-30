from highway_env.envs.common.observation import ObservationType
from gym.spaces import Box, Discrete, Dict
import numpy as np
from .observation_helper import process_obs
from .settings import feature_indices, numHA, laneFinder, lanes_count, motor_model




class MyObservation(ObservationType):
  def __init__(self, env, base_obs):
    super().__init__(env)
    self.base_obs = base_obs
  def space(self):
      return Box(                   low=np.array(
[136.89, -0.15, 18.971249999999998, -0.1, 136.89, 155.1275, 15.70125, 153.665, -0.2025, -23.365000000000002, -0.19875, -18.365000000000002, -0.19875, -22.2475, -0.22999999999999998, -18.365000000000002, -0.10250000000000001, -18.365000000000002]
        ), 
                                  high=np.array(
[594.0975000000001, 31.669999999999998, 50.2475, 0.20625000000000002, 718.25125, 672.09375, 50.14125, 638.8525, 0.17250000000000001, 25.259999999999998, 0.07250000000000001, 25.3825, 0.07250000000000001, 17.759999999999998, 0.07250000000000001, 17.759999999999998, 0.17124999999999999, 17.759999999999998]
        ), 
                                  shape=(18,), dtype=np.float32)
  def observe(self):
    res = process_obs(self.env, self.base_obs.observe())
    # print(res[feature_indices])
    # print(self.get_lane())
    # print(f"crashed   {self.env.vehicle.crashed}   offroad   {not self.env.vehicle.on_road}")
    res = res[feature_indices]
    res = np.append(res, self.env.last_action)
    res = np.append(res, motor_model(res))
    return res
  def get_lane(self):
    return laneFinder(process_obs(self.env, self.base_obs.observe())[1])
