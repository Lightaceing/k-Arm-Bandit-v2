from strategy.base import Strategy
from strategy.exploit import Exploit
from strategy.explore import Explore

import numpy as np


class EpsilonGreedy(Strategy):

    def __init__(self, epsilon_value: float):
        self.name = "Epsilon Greedy"
        self.epsilon_value = epsilon_value
        self.explore = Explore()
        self.exploit = Exploit()

    def select_arm(self, agent):  # TODO:Simplify
        val = np.random.uniform(0, 1)
        if (val <= self.epsilon_value):
            random_arm = self.explore.select_arm(agent)
            return random_arm
        else:
            arm = self.exploit.select_arm(agent)
            return arm.item()
