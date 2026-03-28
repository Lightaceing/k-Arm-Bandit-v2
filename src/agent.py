import numpy as np

arm_count = 10

# Implementing Class for entire learning algo


class LearnAlgo:

    def __init__(self, arm_count=10):
        self.arm_count = arm_count
        self.estimated_rewards = np.zeros(
            shape=arm_count)  # Default is all zeros
        self.counts = np.zeros(shape=arm_count)

        # For graph
        self.tracking_graph = np.zeros(shape=(arm_count, 2))

    def initialize_empty_reward(self):
        self.estimated_rewards = np.zeros(
            shape=arm_count)

    def update_estimate(self, arm_selected, reward):
        self.counts[arm_selected] += 1
        self.estimated_rewards[arm_selected] = self.estimated_rewards[arm_selected] + (
            1/self.counts[arm_selected])*(reward - self.estimated_rewards[arm_selected])

    def epsilon_greedy(self, epsilon=0.2):
        val = np.random.uniform(0, 1)
        if (val <= epsilon):
            arm = self.explore()
        else:
            arm = self.exploit()

        return arm

    def exploit(self):
        arm = np.argmax(self.estimated_rewards)
        return arm

    def explore(self):
        random_arm = np.random.randint(0, self.arm_count)
        return random_arm
