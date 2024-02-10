# N x T x Dim



import pandas as pd
import numpy as np
from settings_j import fname, N_t, obs_headers, acts_headers, mm_headers, ha_headers, train_or_test, motor_model, pv_stddev, train_end, test_end
from scipy.stats import norm

def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)




if train_or_test==0:
  traj_start = 0
  traj_end = train_end
  outname = "multimodal_push"
elif train_or_test==1:
  traj_start = train_end
  traj_end = test_end
  outname = "validation"
elif train_or_test==2:
  traj_start = 0
  traj_end = test_end
  outname = "combined"
else:
  exit()
N_traj = traj_end-traj_start

N_obs = len(obs_headers)+len(acts_headers)+len(mm_headers)
N_acts = len(acts_headers)
N_mm = len(mm_headers)
N_ha = len(ha_headers)


obs_all = np.empty((N_traj, N_t, N_obs))
acts_all = np.empty((N_traj, N_t, N_acts))
ha_all = np.empty((N_traj, N_t, N_ha))
masks_all = np.ones((N_traj, N_t, 1))

for i in range(traj_start, traj_end):
  x = pd.read_csv(f"{fname}/data{i}.csv", delimiter=",")
  for name in x.columns:
    if name.strip() in acts_headers:
      new_name = "prev."+name.strip()
      x[new_name] = x[name].shift(1)
      x.at[0, new_name] = 0.0
      obs_headers.append(new_name)
  obs = np.empty((0,N_t))
  acts = np.empty((0,N_t))
  ha = np.empty((0,N_t))
  for name in x.columns:
    temp = np.expand_dims(x[name].to_numpy(), axis=0)
    if name.strip() in acts_headers:
      acts = np.append(acts, temp, axis=0)
      print(f"ACTS: {name.strip()}")
    elif name.strip() in obs_headers:
      obs = np.append(obs, temp, axis=0)
      print(f"OBS: {name.strip()}")
    elif name.strip() in ha_headers:
      ha = np.append(ha, temp, axis=0)
      print(f"HA: {name.strip()}")

  obs = np.swapaxes(obs, 0, 1)
  acts = np.swapaxes(acts, 0, 1)
  ha = np.swapaxes(ha, 0, 1)

  mm = np.empty((0, N_mm))
  for t in range(N_t):
    temp = np.expand_dims(motor_model(obs[t]), axis=0)
    mm = np.append(mm, temp, axis=0)
  for header in mm_headers:
    print(f"OBS: {header}")
  obs = np.append(obs, mm, axis=1)

  obs_all[i-traj_start] = obs
  acts_all[i-traj_start] = acts
  ha_all[i-traj_start] = ha
  print()




np.set_printoptions(threshold=np.inf, suppress=True)

print(obs_all.shape)
print(acts_all.shape)
print(ha_all.shape)
print()

bounds_min = []
bounds_max = []
for i in range(obs_all.shape[2]):
  m1 = np.min(obs_all[:,:,i])
  m2 = np.max(obs_all[:,:,i])
  bounds_min.append(m1 * 1.25)
  bounds_max.append(m2 * 1.25)

print(bounds_min)
print(bounds_max)


sum_sum_ll = 0.
for i in range(obs_all.shape[0]):
  sum_ll = 0.
  for j in range(obs_all.shape[1]):
    ha = ha_all[i,j,0]
    mm_start = int(N_obs-N_mm+N_acts*ha)
    mm_end = int(N_obs-N_mm+N_acts*(ha+1))
    noiseless_act = obs_all[i,j,mm_start:mm_end]
    noisy_act = acts_all[i,j,:]
    err = np.sum(get_err(noisy_act, noiseless_act, pv_stddev))
    sum_ll += err
  sum_sum_ll += sum_ll
  print(sum_ll / obs_all.shape[1])
print("Overall likelihood of ground truth policy")
print(sum_sum_ll / obs_all.shape[0] / obs_all.shape[1])


np.save(f"{fname}/{outname}_observations", obs_all)
np.save(f"{fname}/{outname}_actions", acts_all)
np.save(f"{fname}/{outname}_ha", ha_all)
np.save(f"{fname}/{outname}_masks", masks_all)


