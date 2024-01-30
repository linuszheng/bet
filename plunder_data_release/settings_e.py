

fname = "e-1d-ss"
N_t = 126
obs_headers = ["pos", "decMax", "accMax", "vMax", "vel", "target"]
acts_headers = ["LA.acc"]
mm_headers = ["mm.acc", "mm.con", "mm.dec"]
ha_headers = ["HA"]
train_or_test = 2
train_end = 10
test_end = 30

pv_stddev = [0.5]


def motor_model(info):
    a = min(info[6] + 1, info[2])
    if info[6] < 0:
        b = min(info[6] + 1, info[2])
    else:
        b = max(info[6] - 1, info[1])
    c = max(info[6] - 1, info[1])
    return [a,b,c]