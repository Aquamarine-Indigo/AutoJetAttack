import numpy as np
DISTANCE_upbound = 120

def check_truncation(my_state, enemy_state):
    flag = False
    curr_dist = np.linalg.norm(enemy_state[0:3] - my_state[0:3])
    if curr_dist >= DISTANCE_upbound:
        flag = True
    return flag