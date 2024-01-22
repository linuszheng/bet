import numpy as np
import gym
from gym import spaces
from .settings import numHA, n_timesteps, pv_stddev, motor_model
from scipy.stats import norm

EPSILON = 10E-10

def get_err(a, b, stdev):
  return np.log(1-norm.cdf(abs(a-b)/stdev))

class LabelStopSignEnv(gym.Env):

    def __init__(self):
      self.obs_all = np.load("/home/linusjz/bet2/bet/plunder_data_release/e-1d-ss/multimodal_push_observations.npy")
      self.acts_all = np.load("/home/linusjz/bet2/bet/plunder_data_release/e-1d-ss/multimodal_push_actions.npy")
      self.t = 0
      self.observation_space = spaces.Box( low=np.array([0, -40, 0, 0, 0, 100]), 
                                  high=np.array([150, 0, 40, 250, 250, 100]), 
                                  shape=(6,), dtype=np.float32)
      self.action_space = spaces.Box(shape=(1,), low=np.array([-40]), high=np.array([40]), dtype=np.float32)
      self.max_t = 125
      self.ll = np.empty((10, 125))
      self.acts = np.empty((10, 125, 1))
      self.n = -1
      self.initial_last_act = [0.]

    def _get_info(self):
      return { }

    def _get_obs(self):
      return self.obs_all[self.n][self.t]

    def _get_prev_act(self):
      if self.t == 0:
        return self.initial_last_act
      else:
        return self.acts_all[self.n][self.t-1]

    def _get_desired_act(self):
      return self.acts_all[self.n][self.t]

    def reset(self, seed=None, options=None):
      self.t = 0
      self.n += 1
      print(self.n)
      return self._get_obs()
    
    def get_nearest(self, possibilities, action):
      rounded_action = [0]
      ll = np.inf
      for p in possibilities:
        ll_temp = abs(action-p)
        if ll_temp < ll:
          rounded_action = p
          ll = ll_temp
      return rounded_action

    def step(self, action):
      possibilities = motor_model(self._get_obs(), self._get_prev_act())
      action = self.get_nearest(possibilities, action)
      ll = get_err(action, self._get_desired_act(), pv_stddev)
      self.ll[self.n][self.t] = ll
      self.acts[self.n][self.t] = action
      self.t += 1
      return self._get_obs(), ll, self.t >= self.max_t, self._get_info()

    def render(self, mode="human"):
      pass
    
    def close(self):
      np.set_printoptions(threshold=np.inf)
      print(self.ll)
      print(self.acts)
      print("avg likelihood")
      print(np.average(self.ll))
      pass
    





gym.register(
    id="labelstopsign-v0",
    entry_point=LabelStopSignEnv
)

