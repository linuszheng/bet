import numpy as np


which_difficulty = 3


lanes_count = 4

pv_range = [
    [-0.3, 0.3],
    [-30, 30]
]




_min_performance_to_save = .64


training_set = 10
validation_set = 30 # including training_set

folders = [
    "a-2d-merge-easy",
    "b-2d-merge-medium",
    "c-2d-merge-hard",
    "d-2d-merge-impossible"
]

all_lows = [
    [136.89, -0.15, 18.971249999999998, -3.4875, -0.1, 136.89, -5.15, 18.189999999999998, -6.59125, -0.35125000000000006, 155.1275, 0.0, 15.70125, -6.603750000000001, -0.34, 153.665, 2.5124999999999997, 15.40625, -5.595, -0.3525, -0.21400356665325576, -25.5069740335993, -0.19766891590028862, -20.5069740335993, -0.19766891590028862, -22.544594518009955, -0.24625000000000002, -20.5069740335993, -0.11400356665325574, -20.5069740335993],
    [136.89, -0.15, 18.971249999999998, -3.4875, -0.1, 136.89, -5.15, 18.189999999999998, -6.59125, -0.35125000000000006, 155.1275, 0.0, 15.70125, -6.603750000000001, -0.34, 153.665, 2.5124999999999997, 15.40625, -5.595, -0.3525, -0.2025, -23.365000000000002, -0.19875, -18.365000000000002, -0.19875, -22.2475, -0.22999999999999998, -18.365000000000002, -0.10250000000000001, -18.365000000000002],
    [136.89, -0.15, 18.971249999999998, -3.4875, -0.1, 136.89, -5.15, 18.189999999999998, -6.59125, -0.35125000000000006, 155.1275, 0.0, 15.70125, -6.603750000000001, -0.34, 153.665, 2.5124999999999997, 15.40625, -5.595, -0.3525, -0.2292338698809279, -26.404724517334618, -0.19813582701645988, -21.404724517334618, -0.19813582701645988, -22.2475, -0.24689470938694014, -21.404724517334618, -0.1292338698809279, -21.404724517334618],
    [136.89, -0.15, 18.971249999999998, -3.4875, -0.1, 136.89, -5.15, 18.189999999999998, -6.59125, -0.35125000000000006, 155.1275, 0.0, 15.70125, -6.603750000000001, -0.34, 153.665, 2.5124999999999997, 15.40625, -5.595, -0.3525, -0.27768363734283963, -25.985190341409915, -0.2198443706616403, -20.985190341409915, -0.2198443706616403, -22.986250000000005, -0.30125, -20.985190341409915, -0.17768363734283962, -20.985190341409915]
]
all_highs = [
    [594.0975000000001, 31.669999999999998, 50.2475, 7.9575, 0.20625000000000002, 718.25125, 24.908749999999998, 50.131249999999994, 7.68, 0.3175, 672.09375, 31.48875, 50.14125, 7.651250000000001, 0.42375, 638.8525, 34.995, 50.2475, 7.92375, 0.37, 0.18135880098092413, 26.069556735218722, 0.08135880098092413, 25.3825, 0.08135880098092413, 18.569556735218722, 0.08135880098092413, 18.569556735218722, 0.17124999999999999, 18.569556735218722],
    [594.0975000000001, 31.669999999999998, 50.2475, 7.9575, 0.20625000000000002, 718.25125, 24.908749999999998, 50.131249999999994, 7.68, 0.3175, 672.09375, 31.48875, 50.14125, 7.651250000000001, 0.42375, 638.8525, 34.995, 50.2475, 7.92375, 0.37, 0.17250000000000001, 25.259999999999998, 0.07250000000000001, 25.3825, 0.07250000000000001, 17.759999999999998, 0.07250000000000001, 17.759999999999998, 0.17124999999999999, 17.759999999999998],
    [594.0975000000001, 31.669999999999998, 50.2475, 7.9575, 0.20625000000000002, 718.25125, 24.908749999999998, 50.131249999999994, 7.68, 0.3175, 672.09375, 31.48875, 50.14125, 7.651250000000001, 0.42375, 638.8525, 34.995, 50.2475, 7.92375, 0.37, 0.19899435713344213, 26.803616813170933, 0.09899435713344214, 25.3825, 0.09899435713344214, 19.303616813170933, 0.09899435713344214, 19.303616813170933, 0.175, 19.303616813170933],
    [594.0975000000001, 31.669999999999998, 50.2475, 7.9575, 0.20625000000000002, 718.25125, 24.908749999999998, 50.131249999999994, 7.68, 0.3175, 672.09375, 31.48875, 50.14125, 7.651250000000001, 0.42375, 638.8525, 34.995, 50.2475, 7.92375, 0.37, 0.19874008204949375, 31.468933367065127, 0.09874008204949375, 24.6275, 0.09874008204949375, 23.968933367065127, 0.09874008204949375, 23.968933367065127, 0.19375, 23.968933367065127]
]


pv_stddevs = [
    [0.005, 0.5],
    [0.01, 1.0],
    [0.02, 2.0],
    [0.03, 3.0]
]


lows = all_lows[which_difficulty]
highs = all_highs[which_difficulty]
pv_stddev = pv_stddevs[which_difficulty]
fn = folders[which_difficulty]


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

