import numpy as np
import gym
from gym import spaces
from .settings import numHA, pv_stddev, motor_model
from scipy.stats import norm

EPSILON = 10E-10
clip = True

def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)

fn = "f-2d-highway"

class LabelHighwayEnv(gym.Env):

    def __init__(self):
      # self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_observations.npy")
      # self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_actions.npy")
      # self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_ha.npy")
      self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_observations.npy")
      self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_actions.npy")
      self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_ha.npy")
      self.N = self.acts_all.shape[0]
      self.t = 0
      self.observation_space = spaces.Box(                   low=np.array(
[175.64875, -2.36875, 19.84625, -11.0175, -0.315, 195.26624999999999, -7.3687499999999995, 14.185, -11.0175, -0.315, 248.6775, 0.0, 13.92125, -7.93625, -0.335, 175.64875, 4.64625, 18.00875, -9.00875, -0.3425, -0.2, -30.19375, -0.19125, -25.19375, -0.19125, -27.372499999999995, -0.2475, -25.19375, -0.15, -25.19375]), 
                                  high=np.array(
[1128.66625, 25.34375, 50.2175, 10.38125, 0.32375, 1227.35625, 20.0, 50.214999999999996, 8.72, 0.31125, 1166.6599999999999, 25.0, 50.214999999999996, 7.60875, 0.32375, 1222.1625, 30.34375, 50.112500000000004, 10.3475, 0.34, 0.16875, 23.41375, 0.17163280049338706, 22.603749999999998, 0.17163280049338706, 15.913750000000002, 0.11875, 15.913750000000002, 0.20125, 15.913750000000002]), 
                                  shape=(30,), dtype=np.float32)
      self.action_space = spaces.Box(shape=(2,), low=np.array([-.3, -30]), high=np.array([.3, 30]), dtype=np.float32)
      self.max_t = self.acts_all.shape[1]
      self.ll = np.empty(self.acts_all.shape[0:2])
      self.acts = np.empty(self.acts_all.shape)
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
      if ll < -10.:
        ll = -10.
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
    id="labelhighway-v0",
    entry_point=LabelHighwayEnv
)

