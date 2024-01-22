# Setting: 1D-target

_min_performance_to_save = .87


training_set = 10
validation_set = 30

n_timesteps = 125

pv_range = [[-50, 50]]
pv_stddev = [0.5]


initialHA = 0
initialLA = [0]


numHA = 3


la_idx_wrt_lim_obs = [6]
la_idx_wrt_all_obs = [1]
features_idx = [0, 1, 2, 3, 4, 5]


def motor_model(obs, la):
    a = [min(la[0] + 1, obs[2])]
    if la[0] < 0:
        b = [min(la[0] + 1, obs[2])]
    else:
        b = [max(la[0] - 1, obs[1])]
    c = [max(la[0] - 1, obs[1])]
    return [a, b, c]

