import numpy as np





fname = "j-highway-human"
N_t = 75
obs_headers = ["x", "y", "vx", "vy", "heading", "l_x", "l_y", "l_vx", "l_vy", "l_heading", "f_x", "f_y", "f_vx", "f_vy", "f_heading", "r_x", "r_y", "r_vx", "r_vy", "r_heading"]
acts_headers = ["LA.acc", "LA.steer"]
mm_headers = ["mm.a1", "mm.a2", "mm.b1", "mm.b2", "mm.c1", "mm.c2", "mm.d1", "mm.d2"]
ha_headers = ["HA"]
train_or_test = 0

train_end = 9
test_end = 18


pv_stddev = [.01, 1]


lane_diff = 4 # Distance lanes are apart from each other
lanes_count = 4 # Number of lanes
use_absolute_lanes = True # Whether or not to label lanes as absolute or relative to current vehicle lane

def laneFinder(y):
    return round(y / lane_diff)

TURN_HEADING = 0.1 # Target heading when turning
TURN_TARGET = 30 # How much to adjust when targeting a lane (higher = smoother)
max_velocity = 25 # Maximum velocity
min_velocity = 15 # Turning velocity



def mm(info, ha):
    global last_action

    target_acc = 0.0
    target_steer = 0.0

    if ha==0:
        # Attain max speed
        target_acc = 4

        # Follow current lane
        target_y = laneFinder(info[1]) * lane_diff
        target_heading = np.arctan((target_y - info[1]) / TURN_TARGET)
        target_steer = max(min(target_heading - info[4], 0.02), -0.02)
    elif ha==1:
        target_acc = -4

        # Follow current lane
        target_y = laneFinder(info[1]) * lane_diff
        target_heading = np.arctan((target_y - info[1]) / TURN_TARGET)
        target_steer = max(min(target_heading - info[4], 0.02), -0.02)
    elif ha==2:
        target_acc = 4
        target_steer = max(min(0.1 - info[4], 0.04), -0.04)
    elif ha==3:
        target_acc = 4
        target_steer = max(min(-0.1 - info[4], 0.04), -0.04)

    if info[2] >= max_velocity - 0.01:
        target_acc = min(target_acc, 0.0)
    if info[2] <= min_velocity + 0.01:
        target_acc = max(target_acc, 0.0)

    if target_steer > 0:
        target_steer = min(target_steer, TURN_HEADING - info[4])
    if target_steer < 0:
        target_steer = max(target_steer, -TURN_HEADING - info[4])

    return [target_steer, target_acc]


def motor_model(info):
    res = mm(info, 0) + mm(info, 1) + mm(info, 2) + mm(info, 3)
    return res



