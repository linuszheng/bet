import numpy as np





pv_stddev = [0.005, 0.5]
# pv_stddev = [0.01, 1.0]
# pv_stddev = [0.02, 2.0]
# pv_stddev = [0.03, 3.0]

lanes_count = 6

pv_range = [
    [-0.3, 0.3],
    [-30, 30]
]




_min_performance_to_save = .64


training_set = 10
validation_set = 30 # including training_set

folder = "merge-easy-data/"
# folder = "merge-medium-data/"
# folder = "merge-hard-data/"
# folder = "merge-impossible-data/"

feature_indices = [0, 1, 2, 4, 5, 10, 12, 15]

pred_var = ["LA.steer", "LA.acc"]

numHA = 4
_n_timesteps = 74
initialHA = 0
initialLA = [0,0]

TURN_HEADING = 0.15 # Target heading when turning
TURN_TARGET = 30 # How much to adjust when targeting a lane (higher = smoother)
MAX_VELOCITY = 45 # Maximum velocity

lane_diff = 4
def laneFinder(y):
    return round(y / lane_diff)



def mm(info, ha):
    last_la = [info[-2], info[-1]]

    target_acc = 0.0
    target_heading = 0.0

    if ha == 0:
        target_acc = MAX_VELOCITY - info[2]

        target_y = laneFinder(info[1]) * 4
        target_heading = np.arctan((target_y - info[1]) / TURN_TARGET)
    elif ha == 1:
        target_acc = info[6] - info[2]   

        target_y = laneFinder(info[1]) * 4
        target_heading = np.arctan((target_y - info[1]) / TURN_TARGET)
    elif ha == 2:
        target_acc = -0.5
        target_heading = -TURN_HEADING
    else:
        target_acc = -0.5
        target_heading = TURN_HEADING

    target_steer = target_heading - info[3]

    if target_steer > last_la[0]:
        target_steer = min(target_steer, last_la[0] + 0.08)
    else:
        target_steer = max(target_steer, last_la[0] - 0.08)

    if target_acc > last_la[1]:
        target_acc = min(target_acc, last_la[1] + 4)
    else:
        target_acc = max(target_acc, last_la[1] - 6)
    return [target_steer, target_acc]

def motor_model(info):
    res = mm(info, 0) + mm(info, 1) + mm(info, 2) + mm(info, 3)
    return res

