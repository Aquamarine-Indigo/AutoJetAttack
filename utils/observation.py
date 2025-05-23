import numpy as np


def marshal_observation(my_state, enemy_state):
    """
    Process the observation from the environment.

    Args:
        my_state (numpy.ndarray): The state of our aircraft, including position, velocity, and attitude.
        enemy_state (numpy.ndarray): The state of the enemy aircraft, including position, velocity, and attitude.

    Returns:
        numpy.ndarray: The processed observation.
    """
    agent_state = np.zeros(shape=[22], dtype=np.float64)
    
    # my state
    agent_state[0:3] = my_state[0:3] / 1000.0  # (x, y, z)
    agent_state[3:6] = my_state[3:6]           # (roll, pitch, yaw)
    agent_state[6:9] = my_state[6:9] / 100.0   # (u, v, w)
    
    # enemy state
    agent_state[9:12] = enemy_state[0:3] / 1000.0  # (x, y, z)
    agent_state[12:15] = enemy_state[3:6]          # (roll, pitch, yaw)
    agent_state[15:18] = enemy_state[6:9] / 100.0  # (u, v, w)
    
    # health
    agent_state[18] = enemy_state[12] / 100.0  
    
    # relative position
    rel_pos = enemy_state[0:3] - my_state[0:3]
    agent_state[19:22] = rel_pos / 1000.0
    
    return agent_state