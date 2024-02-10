import numpy as np





pv_stddev = [.01, 2]

_n_timesteps = 150

numHA = 4

KP_H = 0.5 # Turning rate
TURN_HEADING = 0.15 # Target heading when turning
TURN_TARGET = 30 # How much to adjust when targeting a lane (higher = smoother)
MAX_VELOCITY = 40 # Maximum velocity
turn_velocity = 30 # Turning velocity

lows = [175.64875, -2.36875, 19.84625, -11.0175, -0.315, 195.26624999999999, -7.3687499999999995, 14.185, -11.0175, -0.315, 248.6775, 0.0, 13.92125, -7.93625, -0.335, 175.64875, 4.64625, 18.00875, -9.00875, -0.3425, -0.2, -30.19375, -0.19125, -25.19375, -0.19125, -27.372499999999995, -0.2475, -25.19375, -0.15, -25.19375]
highs = [1128.66625, 25.34375, 50.2175, 10.38125, 0.32375, 1227.35625, 20.0, 50.214999999999996, 8.72, 0.31125, 1166.6599999999999, 25.0, 50.214999999999996, 7.60875, 0.32375, 1222.1625, 30.34375, 50.112500000000004, 10.3475, 0.34, 0.16875, 23.41375, 0.17163280049338706, 22.603749999999998, 0.17163280049338706, 15.913750000000002, 0.11875, 15.913750000000002, 0.20125, 15.913750000000002]

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

