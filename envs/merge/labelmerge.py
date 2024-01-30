import numpy as np
import gym
from gym import spaces
from .settings import numHA, pv_stddev, motor_model
from scipy.stats import norm

EPSILON = 10E-10
clip = True

def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)

fn = "a-2d-merge-easy"

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
      self.observation_space = spaces.Box(                   low=np.array(
[136.89, -0.15, 18.971249999999998, -3.4875, -0.1, 136.89, -5.15, 18.189999999999998, -6.59125, -0.35125000000000006, 155.1275, 0.0, 15.70125, -6.603750000000001, -0.34, 153.665, 2.5124999999999997, 15.40625, -5.595, -0.3525, -0.2025, -23.365000000000002, -0.19875, -18.365000000000002, -0.19875, -22.2475, -0.22999999999999998, -18.365000000000002, -0.10250000000000001, -18.365000000000002]), 
                                  high=np.array(
[594.0975000000001, 31.669999999999998, 50.2475, 7.9575, 0.20625000000000002, 718.25125, 24.908749999999998, 50.131249999999994, 7.68, 0.3175, 672.09375, 31.48875, 50.14125, 7.651250000000001, 0.42375, 638.8525, 34.995, 50.2475, 7.92375, 0.37, 0.17250000000000001, 25.259999999999998, 0.07250000000000001, 25.3825, 0.07250000000000001, 17.759999999999998, 0.07250000000000001, 17.759999999999998, 0.17124999999999999, 17.759999999999998]), 
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
    id="labelmerge-v0",
    entry_point=LabelMergeEnv
)

