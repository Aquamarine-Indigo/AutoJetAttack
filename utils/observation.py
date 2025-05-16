import numpy as np
from scipy.spatial.transform import Rotation as R

# This is the observation processing function. Remember to modify the declarations in trainenv.py correspondingly.
def marshal_observation(my_state, enemy_state):
#     agent_state = np.zeros(shape=[15], dtype=np.float64)
    my_xyz = my_state[:3]
    enemy_xyz = enemy_state[:3]
    my_euler_angle = my_state[3:6]
    
    my_rotation = R.from_euler('xyz', my_euler_angle)
    heading_vector = my_rotation.apply([1, 0, 0])
    target_vector = enemy_xyz - my_xyz
    target_direction = target_vector / np.linalg.norm(target_vector)
    cos_theta = np.dot(heading_vector, target_direction)
    angle_diff = np.arccos(np.clip(cos_theta, -1.0, 1.0)).reshape(1)
    
    # print(my_state.shape, enemy_state.shape, target_vector.shape, angle_diff.shape)
    
    agent_state = np.concatenate((my_state, enemy_state, target_vector, angle_diff)) # shape 26 + 3 + 1
    return agent_state