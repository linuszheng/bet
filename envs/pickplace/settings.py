import numpy as np




pv_stddev = [0.2, 0.2, 0.2, 0.2]
numHA = 4


pv_range = [
    [-2, 2],
    [-2, 2],
    [-2, 2],
    [-2, 2],
]



def mm(info, ha):
    data = info
    data_prev = info
    bx, by, bz = data[3] - data[0], data[4] - data[1], data[5] - data[2]
    
    if ha == 0:
        return [bx * 5.0, by * 5.0, bz * 5.0, 0.6]
    
    if data_prev[13] >= 0.5:
        return [bx * 5.0, by * 5.0, bz * 5.0, 0.3]
    elif data_prev[13] >= 0.3:
        return [bx * 5.0, by * 5.0, bz * 5.0, 0]
    elif data_prev[13] >= 0.1:
        return [bx * 5.0, by * 5.0, bz * 5.0, -0.3]
    elif data_prev[13] >= -0.1:
        return [bx * 5.0, by * 5.0, bz * 5.0, -0.6]
    
    vx = 5 * (data[6] - data[0])
    vy = 5 * (data[7] - data[1])
    vz = 5 * (data[8] - data[2])
    end = -0.6
    
    return [vx, vy, vz, end]



def motor_model(info):
    res = mm(info, 0) + mm(info, 1)
    return res

