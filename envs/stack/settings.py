import numpy as np






numHA = 5

pv_stddev = [0.3, 0.3, 0.3, 0.3]

n_timesteps = 150


def mm(info, ha):

    data = info
    data_prev = info

    bx1, by1, bz1, bx2, by2, bz2 = data[4], data[5], data[6], data[7], data[8], data[9]
    tx2, ty2, tz2 = data[13], data[14], data[15]

    tz2 += 0.01

    if ha == 0:
        action = [bx1 * 4.0, by1 * 4.0, bz1 * 4.0, 1]
    elif ha == 1:
        action = [tx2 * 4.0, ty2 * 4.0, tz2 * 4.0, -1]
        if action[2] < 0:
            action[2] = max(data_prev[18] - 0.15, action[2])
    elif ha == 2:
        action = [0, 0, 0.5, 1]
    elif ha == 3:
        action = [bx2 * 4.0, by2 * 4.0, bz2 * 4.0, 1]
        if action[2] < 0:
            action[2] = max(data_prev[18] - 0.15, action[2])
    elif ha == 4:
        action = [0, 0, 0.5, -1]

    return action


def motor_model(info):
    res = mm(info, 0) + mm(info, 1) + mm(info, 2) + mm(info, 3) + mm(info, 4)
    return res
