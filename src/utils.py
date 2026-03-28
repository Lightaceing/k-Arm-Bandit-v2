import numpy as np


def pull_arm(enviroment, arm_selected, s_d=1):
    """
    Selects an arm based on the tech. given by select_arm fn. and generates a reward based on the arm's mean selected

    """
    arm_mean = enviroment[arm_selected]
    reward = np.random.normal(loc=arm_mean, scale=s_d)

    return reward
