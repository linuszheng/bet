import numpy as np





pv_stddev = [.01, 2]

numHA = 4

KP_H = 0.5 # Turning rate
TURN_HEADING = 0.15 # Target heading when turning
TURN_TARGET = 30 # How much to adjust when targeting a lane (higher = smoother)
MAX_VELOCITY = 40 # Maximum velocity
turn_velocity = 30 # Turning velocity

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
        target_acc = turn_velocity - info[2]
        target_heading = -TURN_HEADING
    else:
        target_acc = turn_velocity - info[2]
        target_heading = TURN_HEADING

    target_steer = target_heading - info[4]

    if target_steer > last_la[0]:
        target_steer = min(target_steer, last_la[0] + 0.04)
    else:
        target_steer = max(target_steer, last_la[0] - 0.04)

    if target_acc > last_la[1]:
        target_acc = min(target_acc, last_la[1] + 4)
    else:
        target_acc = max(target_acc, last_la[1] - 6)
    return [target_steer, target_acc]

def motor_model(info):
    res = mm(info, 0) + mm(info, 1) + mm(info, 2) + mm(info, 3)
    return res

