# multimodal_push_observations
# multimodal_push_actions
# multimodal_push_masks
# N x T x Dim

import pandas as pd
import numpy as np

# fname = "a-2d-merge-easy"
# fname = "b-2d-merge-medium"
# fname = "c-2d-merge-hard"
# fname = "d-2d-merge-impossible"
# N_traj = 30
# N_t = 75
# obs_headers = ["x", "vx", "l_x", "f_x", "r_x"]
# acts_headers = ["LA.acc", "LA.steer"]

# fname = "e-1d-ss"
# N_traj = 30
# N_t = 126
# obs_headers = ["pos", "decMax", "accMax", "vMax", "vel", "target"]
# acts_headers = ["LA.acc"]





N_obs = len(obs_headers)
N_acts = len(acts_headers)
N_masks = 1



obs_all = np.empty((N_traj, N_t, N_obs))
acts_all = np.empty((N_traj, N_t, N_acts))
masks_all = np.ones((N_traj, N_t, N_masks))

for i in range(N_traj):
  x = pd.read_csv(f"{fname}/data{i}.csv", delimiter=",")
  obs = np.empty((0,N_t))
  acts = np.empty((0,N_t))
  masks = np.empty((0,N_t))
  for name in x.columns:
    if name.strip() in acts_headers:
      temp = np.expand_dims(x[name].to_numpy(), axis=0)
      acts = np.append(acts, temp, axis=0)
      print(f"ACTS: {name.strip()}")
    elif name.strip() in obs_headers:
      temp = np.expand_dims(x[name].to_numpy(), axis=0)
      obs = np.append(obs, temp, axis=0)
      print(f"OBS: {name.strip()}")
  print()
  obs = np.swapaxes(obs, 0, 1)
  acts = np.swapaxes(acts, 0, 1)
  obs_all[i] = obs
  acts_all[i] = acts


np.save(f"{fname}/multimodal_push_observations", obs_all)
np.save(f"{fname}/multimodal_push_actions", acts_all)
np.save(f"{fname}/multimodal_push_masks", masks_all)
