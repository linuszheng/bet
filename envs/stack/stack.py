import numpy as np
import gym
from gym import spaces
from .settings import numHA, n_timesteps, pv_stddev, motor_model
from panda_gym.envs import PandaStackEnv
from gym.utils import seeding

EPSILON = 10E-10
obs_min = [-0.189755425, -0.19257195000000002, 0.028373855000000003, -1.4782155e-08, -0.242839525, -0.34875512499999994, -0.232565775, -0.31419984999999995, -0.32352380000000003, -0.2250856375, -0.10575550624999999, -0.1874311125, 0.025, -0.25865425000000003, -0.336115025, -0.1825528625, -1.25, -1.25, -1.25, -1.25, -0.9713581, -1.3950204999999998, -0.9302631, 1.25, -1.0346170000000001, -1.3444601, -0.65950775, -1.25, 0.0, 0.0, 0.625, 1.25, -1.2567993999999998, -1.2940952000000001, -0.8240253635469048, 1.25, 0.0, 0.0, 0.625, -1.25]
obs_max = [0.22607428749999997, 0.16939608750000001, 0.2575528625, 0.10030173749999999, 0.2999915875, 0.2471130875, 0.022761425, 0.31384653749999997, 0.33517875, -0.0032965537500000003, 0.1591309125, 0.12675881249999998, 0.025, 0.31676407500000003, 0.3121527625, 0.046626143749999995, 1.25, 1.25, 1.25, 1.25, 1.19996635, 0.98845235, 0.0910457, 1.25, 1.2670563000000001, 1.24861105, 1.0625, -1.25, 0.0, 0.0, 0.625, 1.25, 1.2553861499999999, 1.340715, 1.0625, 1.25, 0.0, 0.0, 0.625, -1.25]


class StackEnv(gym.Env):

  panda_env = PandaStackEnv()

  def __init__(self):
    self.observation_space = spaces.Box(    shape=(40,),               low=np.array(obs_min), high=np.array(obs_max), dtype=np.float32)
    self.action_space = spaces.Box(shape=(4,), low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]), dtype=np.float32)
    self.ep_is_success = False
    self.n_success = 0
    self.reset()
    
  
  def reset(self):
    if self.ep_is_success:
      self.ep_is_success = False
      self.n_success += 1
    print(self.n_success)
    self.panda_env.reset()
    self.t = 0
    self.last_la = [0, 0, 0, 0]
    return self._get_obs()

  def _get_obs(self):
    observation = self.panda_env._get_obs()
    
    
    world_state = observation["observation"]
    target_bottom = observation["desired_goal"][0:3]
    target_top = observation["desired_goal"][3:6]

    x, y, z, end_width = world_state[0], world_state[1], world_state[2], world_state[6]
    bx1, by1, bz1, bx2, by2, bz2 = world_state[7], world_state[8], world_state[9], world_state[19], world_state[20], world_state[21]
    tx1, ty1, tz1, tx2, ty2, tz2 = target_bottom[0], target_bottom[1], target_bottom[2], target_top[0], target_top[1], target_top[2]

    bx1, by1, bz1, bx2, by2, bz2 = bx1 - x, by1 - y, bz1 - z, bx2 - x, by2 - y, bz2 - z
    tx2, ty2, tz2 = tx2 - x, ty2 - y, tz2 - z

    obs_pruned = [x, y, z, end_width, bx1, by1, bz1, bx2, by2, bz2, tx1, ty1, tz1, tx2, ty2, tz2]
        
    obs_pruned = np.concatenate((obs_pruned, self.last_la))
    return np.concatenate((obs_pruned, motor_model(obs_pruned)))

  def _get_info(self):
    return {}

  def step(self, la_to_take):
    self.t += 1
    self.ep_is_success = self.panda_env.step(la_to_take)[2]
    self.last_la = la_to_take
    return self._get_obs(), 0, self.t > n_timesteps, {}
  
  def render(self, mode="human"):
    pass


gym.register(
    id="stack-v0",
    entry_point=StackEnv
)

