

fname = "g-pick-place-policy"
N_t = 50
obs_headers = ["x", "y", "z", "bx", "by", "bz", "tx", "ty", "tz", "end_width"]

acts_headers = ["LA.vx", "LA.vy", "LA.vz", "LA.end"]
mm_headers = ["mm.a1", "mm.a2", "mm.a3", "mm.a4", "mm.b1", "mm.b2", "mm.b3", "mm.b4"]
ha_headers = ["HA"]
train_or_test = 2
train_end = 10
test_end = 20




pv_stddev = [0.2, 0.2, 0.2, 0.2]



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

