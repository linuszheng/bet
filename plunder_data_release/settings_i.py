
fname = "i-stack"
N_t = 150
obs_headers = ["x", "y", "z", "end_width", "bx1", "by1", "bz1", "bx2", "by2", "bz2", "tx1", "ty1", "tz1", "tx2", "ty2", "tz2"]

acts_headers = ["LA.vx", "LA.vy", "LA.vz", "LA.end"]
mm_headers = ["mm.a1", "mm.a2", "mm.a3", "mm.a4", 
              "mm.b1", "mm.b2", "mm.b3", "mm.b4",
              "mm.c1", "mm.c2", "mm.c3", "mm.c4",
              "mm.d1", "mm.d2", "mm.d3", "mm.d4",
              "mm.e1", "mm.e2", "mm.e3", "mm.d4",]
ha_headers = ["HA"]
train_or_test = 2
train_end = 10
test_end = 20



pv_stddev = [0.3, 0.3, 0.3, 0.3]




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

