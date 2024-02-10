import numpy as np
import gym
from gym import spaces
from .settings import numHA, pv_stddev, motor_model, fn, lows, highs
from scipy.stats import norm

EPSILON = 10E-10
clip = True

def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)

ll_lim = -10.

class LabelMergeEnv(gym.Env):

    def __init__(self):
      self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_observations.npy")
      self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_actions.npy")
      self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_ha.npy")
      # self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_observations.npy")
      # self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_actions.npy")
      # self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_ha.npy")
      self.N = self.acts_all.shape[0]
      self.t = 0
      self.observation_space = spaces.Box(low=np.array(lows), high=np.array(highs), shape=(30,), dtype=np.float32)
      self.action_space = spaces.Box(shape=(2,), low=np.array([-.3, -30]), high=np.array([.3, 30]), dtype=np.float32)
      self.max_t = self.acts_all.shape[1]
      self.ll = np.zeros(self.acts_all.shape[0:2])
      self.acts = np.zeros(self.acts_all.shape)
      self.n = -1
      self.initial_last_act = np.zeros(self.acts_all.shape[1])

    def _get_info(self):
      return { }

    def _get_obs(self):
      return self.obs_all[self.n][self.t]


    def _get_desired_act(self):
      return self.acts_all[self.n][self.t]

    def reset(self, seed=None, options=None):
      if self.n >= 0:
        print(np.average(self.ll[self.n]))
      self.t = 0
      self.n += 1
      self.n %= self.N
      print(self.n)
      return self._get_obs()
    
    def step(self, action):
      desired_action = self._get_desired_act()
      other_options = self._get_obs()[-8:]
      opt1 = get_err(action, other_options[0:2], pv_stddev).sum()
      opt2 = get_err(action, other_options[2:4], pv_stddev).sum()
      opt3 = get_err(action, other_options[4:6], pv_stddev).sum()
      opt4 = get_err(action, other_options[6:8], pv_stddev).sum()
      opt_errors = np.array([opt1, opt2, opt3, opt4])
      action_i = np.argmax(opt_errors)
      action = other_options[(2*action_i) : (2*action_i+2)]
      ll = np.sum(get_err(action, desired_action, pv_stddev))
      if ll < ll_lim:
        ll = ll_lim
      self.ll[self.n][self.t] = ll
      self.acts[self.n][self.t] = action
      self.t += 1
      return self._get_obs(), ll, self.t >= self.max_t-1, self._get_info()

    def render(self, mode="human"):
      pass
    
    def close(self):
      np.set_printoptions(threshold=np.inf, suppress=True)
      output = np.append(np.expand_dims(self.ll, axis=2), self.acts, axis=2)
      output = np.append(output, self.acts_all, axis=2)
      print(output)
      print("avg likelihood")
      print(np.average(self.ll))
      pass
    






gym.register(
    id="labelmerge-v0",
    entry_point=LabelMergeEnv
)

