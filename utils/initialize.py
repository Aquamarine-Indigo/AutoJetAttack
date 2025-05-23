import numpy as np


def generate_initial_state():
    # my_initial_state = np.zeros(12)
    # enemy_initial_state = np.ones(12)

    my_initial_state = np.array([
        0.0, 0.0, 0.0,   #  (x, y, z)
        0.0, 0.0, 0.0,   #  (roll, pitch, yaw)
        0.0, 0.0, 0.0,    #  (vx, vy, vz)
        0.0, 0.0, 0.0
    ], dtype=np.float64)

    enemy_initial_state = np.array([
        100.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], dtype=np.float64)


    initial_state = np.append(my_initial_state, enemy_initial_state)
    return initial_state