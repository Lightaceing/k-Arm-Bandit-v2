import numpy as np


def create_enviroment(arm_count, max_mean):
    """
    Creates K Arm Bandit Enviroment
    Returns a dictionary with arm no. and corresponding set mean.
    This is used to generate a new reward each time an arm is selected.

    arm_count        : no. of arms
    max_mean         : maximum mean value of an arm
    """
    enviroment = {}

    for i in range(0, arm_count):
        # Generating random value for mean
        random_mean = np.random.uniform(0, max_mean)
        enviroment[i] = random_mean

    return enviroment
