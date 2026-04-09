from strategy.base import Strategy

import numpy as np


class Explore(Strategy):

    def __init__(self):
        self.name = "Explore"

    def select_arm(self, agent):
        # Selects the arm randomly
        random_arm = np.random.randint(0, agent.environment.arm_count)
        return random_arm
