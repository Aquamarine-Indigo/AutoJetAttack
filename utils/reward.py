# This is the reward calculation function. We provide current state and previous state for you.
from config.rewards import RewardsConfig
import numpy as np
from scipy.spatial.transform import Rotation as R

cfg = RewardsConfig()

def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    my_xyz = my_state[:3]
    enemy_xyz = enemy_state[:3]
    my_euler_angle = my_state[3:6]
    enemy_euler_angle = enemy_state[3:6]
    distance_reward = cfg.distance_reward_scale * np.exp(-np.linalg.norm(my_xyz - enemy_xyz) / cfg.distance_reward_sigma)
    
    my_rotation = R.from_euler('xyz', my_euler_angle)
    heading_vector = my_rotation.apply([1, 0, 0])
    target_vector = enemy_xyz - my_xyz
    target_direction = target_vector / np.linalg.norm(target_vector)
    cos_theta = np.dot(heading_vector, target_direction)
    angle_diff = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_reward = cfg.angle_reward_scale * np.exp(-angle_diff / cfg.angle_reward_sigma)
    
    my_blood = my_state[-1]
    enemy_blood = enemy_state[-1]
    
    reward = distance_reward + angle_reward + my_blood*cfg.my_blood_reward + enemy_blood*cfg.enemy_blood_reward
    
    string = f"""
Values:
    My XYZ: {my_xyz}
    Enemy distance: {np.linalg.norm(my_xyz - enemy_xyz)}
    My Euler Angle: {my_euler_angle}
    Enemy Euler Angle: {enemy_euler_angle}
    Distance: {np.linalg.norm(my_xyz - enemy_xyz)}
    Cosine of Angle: {cos_theta}
    Angle Difference: {angle_diff}
    My Blood: {my_blood}
    Enemy Blood: {enemy_blood}
Rewards: 
    Distance Reward: {distance_reward}
    Angle Reward: {angle_reward}
    My Blood Reward: {my_blood*cfg.my_blood_reward}
    Enemy Blood Reward: {enemy_blood*cfg.enemy_blood_reward}
    Total Reward: {reward}
    """
    # print(string)
    # print(reward.shape)
    
    return reward, string