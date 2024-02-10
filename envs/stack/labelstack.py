import numpy as np
import gym
from gym import spaces
from .settings import numHA, pv_stddev, motor_model
from scipy.stats import norm

EPSILON = 10E-10
clip = True

def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)

fn = "i-stack"

obs_min = [-0.189755425, -0.19257195000000002, 0.028373855000000003, -1.4782155e-08, -0.242839525, -0.34875512499999994, -0.232565775, -0.31419984999999995, -0.32352380000000003, -0.2250856375, -0.10575550624999999, -0.1874311125, 0.025, -0.25865425000000003, -0.336115025, -0.1825528625, -1.25, -1.25, -1.25, -1.25, -0.9713581, -1.3950204999999998, -0.9302631, 1.25, -1.0346170000000001, -1.3444601, -0.65950775, -1.25, 0.0, 0.0, 0.625, 1.25, -1.2567993999999998, -1.2940952000000001, -0.8240253635469048, 1.25, 0.0, 0.0, 0.625, -1.25]
obs_max = [0.22607428749999997, 0.16939608750000001, 0.2575528625, 0.10030173749999999, 0.2999915875, 0.2471130875, 0.022761425, 0.31384653749999997, 0.33517875, -0.0032965537500000003, 0.1591309125, 0.12675881249999998, 0.025, 0.31676407500000003, 0.3121527625, 0.046626143749999995, 1.25, 1.25, 1.25, 1.25, 1.19996635, 0.98845235, 0.0910457, 1.25, 1.2670563000000001, 1.24861105, 1.0625, -1.25, 0.0, 0.0, 0.625, 1.25, 1.2553861499999999, 1.340715, 1.0625, 1.25, 0.0, 0.0, 0.625, -1.25]

class LabelStackEnv(gym.Env):

    def __init__(self):
      self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_observations.npy")
      self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_actions.npy")
      self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/validation_ha.npy")
      # self.obs_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_observations.npy")
      # self.acts_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_actions.npy")
      # self.ha_all = np.load(f"/home/linusjz/bet2/bet/plunder_data_release/{fn}/multimodal_push_ha.npy")
      self.N = self.acts_all.shape[0]
      self.t = 0
      self.observation_space = spaces.Box(    shape=(40,),               low=np.array(obs_min), high=np.array(obs_max), dtype=np.float32)
      self.action_space = spaces.Box(shape=(4,), low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]), dtype=np.float32)
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
    id="labelstack-v0",
    entry_point=LabelStackEnv
)

