import numpy as np
import gym
from gym import spaces
from .settings import numHA, n_timesteps, pv_stddev, motor_model
from scipy.stats import norm

EPSILON = 10E-10
clip = True

def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)

class LabelStopSignEnv(gym.Env):

    def __init__(self):
      # self.obs_all = np.load("/home/linusjz/bet2/bet/plunder_data_release/e-1d-ss/multimodal_push_observations.npy")
      # self.acts_all = np.load("/home/linusjz/bet2/bet/plunder_data_release/e-1d-ss/multimodal_push_actions.npy")
      self.obs_all = np.load("/home/linusjz/bet2/bet/plunder_data_release/e-1d-ss/validation_observations.npy")
      self.acts_all = np.load("/home/linusjz/bet2/bet/plunder_data_release/e-1d-ss/validation_actions.npy")
      self.ha_all = np.load("/home/linusjz/bet2/bet/plunder_data_release/e-1d-ss/validation_ha.npy")
      self.N = len(self.acts_all)
      self.t = 0
      self.observation_space = spaces.Box( low=np.array([0, -40, 0, 0, 0, 100] + [-40]*4), 
                                  high=np.array([150, 0, 40, 250, 250, 100] + [40]*4), 
                                  shape=(10,), dtype=np.float32)
      self.action_space = spaces.Box(shape=(1,), low=np.array([-40]), high=np.array([40]), dtype=np.float32)
      self.max_t = 126
      self.ll = np.empty((self.N, 126))
      self.acts = np.empty((self.N, 126, 1))
      self.n = -1
      self.initial_last_act = [0.]
      self.acc = 0
      self.acc_total = 0

    def _get_info(self):
      return { }

    def _get_obs(self):
      if self.t > self.max_t-1:
        return self.obs_all[self.n][self.max_t-1]
      return self.obs_all[self.n][self.t]

    def _get_prev_act(self):
      if self.t == 0:
        return self.initial_last_act
      else:
        return self.acts_all[self.n][self.t-1]

    def _get_desired_act(self):
      return self.acts_all[self.n][self.t]

    def reset(self, seed=None, options=None):
      if self.n >= 0:
        print(np.average(self.ll[self.n]))
      self.t = 0
      print(self.n)
      print(self.acc/self.max_t)
      self.acc = 0
      self.n += 1
      self.n %= self.N
      return self._get_obs()
    
    def get_nearest(self, possibilities, action):
      ll = np.inf
      for (i, p) in enumerate(possibilities):
        ll_temp = abs(action-p)
        if ll_temp < ll:
          rounded_action = p
          idx = i
          ll = ll_temp
      return idx, rounded_action

    def step(self, action):
      possibilities = motor_model(self._get_obs())
      if clip:
        idx, action = self.get_nearest(possibilities, action)
      ll = get_err(action, self._get_desired_act(), pv_stddev)
      self.ll[self.n][self.t] = ll
      self.acts[self.n][self.t] = action
      if idx == self.ha_all[self.n][self.t]:
        self.acc += 1.
        self.acc_total += 1.
      self.t += 1
      return self._get_obs(), ll, self.t >= self.max_t, self._get_info()

    def render(self, mode="human"):
      pass
    
    def close(self):
      np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
      output = np.append(np.expand_dims(self.ll, axis=2), self.acts, axis=2)
      output = np.append(output, self.acts_all, axis=2)
      print(output)
      print("avg likelihood")
      print(np.average(self.ll))
      print("avg accuracy")
      print(self.acc_total/self.N/self.max_t)
      pass
    





gym.register(
    id="labelstopsign-v0",
    entry_point=LabelStopSignEnv
)

