import numpy as np


# fname = "a-2d-merge-easy"
fname = "b-2d-merge-medium-large"
# fname = "b-2d-merge-medium"
# fname = "c-2d-merge-hard"
# fname = "d-2d-merge-impossible"
N_t = 75
obs_headers = ["x", "y", "vx", "vy", "heading", "l_x", "l_y", "l_vx", "l_vy", "l_heading", "f_x", "f_y", "f_vx", "f_vy", "f_heading", "r_x", "r_y", "r_vx", "r_vy", "r_heading"]
acts_headers = ["LA.acc", "LA.steer"]
mm_headers = ["mm.a1", "mm.a2", "mm.b1", "mm.b2", "mm.c1", "mm.c2", "mm.d1", "mm.d2"]
ha_headers = ["HA"]
train_or_test = 0
train_end = 30
test_end = 30

# pv_stddev = [0.005, 0.5]
pv_stddev = [0.01, 1.0]
# pv_stddev = [0.02, 2.0]
# pv_stddev = [0.03, 3.0]



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
        target_acc = info[12] - info[2]   

        target_y = laneFinder(info[1]) * 4
        target_heading = np.arctan((target_y - info[1]) / TURN_TARGET)
    elif ha == 2:
        target_acc = -0.5
        target_heading = -TURN_HEADING
    else:
        target_acc = -0.5
        target_heading = TURN_HEADING

    target_steer = target_heading - info[4]

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

