import numpy as np
import gym
from gym import spaces
from .settings import numHA, pv_stddev, motor_model
from scipy.stats import norm

EPSILON = 10E-10
clip = True

def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)

fn = "g-pick-place-policy"

obs_min = [-0.14771309999999999, -0.23160631250000002, 0.0048890291249999995, -0.1527093125, -0.24460781250000002, 0.024538135000000003, -0.11967610625, -0.160605425, 0.025, 4.29890925e-08, -1.2241278620738945, -1.1626909868023398, -1.25, -1.25, -1.002689975, -0.908584359, -1.1088294125, 0.75, -1.3817946, -1.01701121875, -1.1088294125, -0.75]
obs_max = [0.17073798750000002, 0.1744322625, 0.2467495375, 0.174665675, 0.2001017, 0.1890869125, 0.1738784625, 0.175174, 0.1727453, 0.100059075, 1.097475279687433, 1.25, 0.8956591423000597, 1.25, 0.6145945625, 0.5118701035, 0.10485015624999999, 0.75, 0.6145945625, 1.586372625, 0.8245543962500002, 0.375]

class LabelPickPlaceEnv(gym.Env):

    def __init__(self):
      self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_observations.npy")
      self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_actions.npy")
      self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_ha.npy")
      # self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_observations.npy")
      # self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_actions.npy")
      # self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_ha.npy")
      self.N = self.acts_all.shape[0]
      self.t = 0
      self.observation_space = spaces.Box(low=np.array(obs_min), 
                                  high=np.array(obs_max), 
                                  shape=(22,), dtype=np.float32)
      self.action_space = spaces.Box(shape=(4,), low=np.array([-2,-2,-2,-2]), high=np.array([2,2,2,2]), dtype=np.float32)
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
      ll = np.sum(get_err(action, desired_action, pv_stddev))
      if ll < -100.:
        ll = -100.
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
    id="labelpickplace-v0",
    entry_point=LabelPickPlaceEnv
)

