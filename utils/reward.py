import numpy as np
import math

DISTANCE_WEIGHT = 5
ANGLE_WEIGHT = 5
HIT_REWARD = 10
# ANGLE_PENALTY = 5
# TIME_PENALTY = -1 

# This is the reward calculation function. We provide current state and previous state for you.
def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    reward = 0.0

    # distance reward
    prev_dist = np.linalg.norm(prev_enemy_state[0:3] - prev_my_state[0:3])
    curr_dist = np.linalg.norm(enemy_state[0:3] - my_state[0:3])
    dist_diff = prev_dist - curr_dist
    # print(dist_diff)
    reward += DISTANCE_WEIGHT * dist_diff

    my_pos = my_state[0:3]
    enemy_pos = enemy_state[0:3]
    direction_to_enemy = enemy_pos - my_pos
    if np.linalg.norm(direction_to_enemy) > 0:
        direction_to_enemy = direction_to_enemy / np.linalg.norm(direction_to_enemy)

    roll, pitch, yaw = my_state[3:6]
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    R_pitch = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    R_yaw = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = R_yaw @ R_pitch @ R_roll
    current_direction = R @ np.array([1, 0, 0])

    # angle reward
    angle_cos = np.clip(np.dot(direction_to_enemy, current_direction), -1.0, 1.0)
    reward += ANGLE_WEIGHT * angle_cos

    # hit reward
    if enemy_state[-1] < prev_enemy_state[-1]:
        health_diff = prev_enemy_state[-1] - enemy_state[-1]
        reward += HIT_REWARD * health_diff

    # # time penalty
    # reward += TIME_PENALTY

    return reward