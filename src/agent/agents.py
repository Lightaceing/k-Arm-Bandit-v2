import numpy as np


class Agent():

    """
    Refactored Agent class



    """

    def __init__(self, environment, strategy):

        # Attribtues taken from enviroment object
        self.environment = environment

        # Taken from strategy
        self.strategy = strategy

        # New attributes defined
        self.estimated_rewards = np.zeros(
            self.environment.arm_count)  # By default defined as 0
        # self.past_strategy_used = {strategy: 0}
        self.actions_taken = []
        self.epochs_taken = 0
        self.optimal_action_history = []
        # Declared for update_estimate
        self.counts = np.zeros(shape=environment.arm_count)
        self.reward_history = []

        self.reward_history_per_arm = []
        for _ in range(self.environment.arm_count):
            self.reward_history_per_arm.append([])

        self.estimate_history = []
        # Creating history
        for _ in range(self.environment.arm_count):
            self.estimate_history.append([])

    def change_strategy(self, new_strategy):
        self.strategy = new_strategy

    def select_policy(self,):
        pass

    def select_arm(self):
        return self.strategy.select_arm(self)

    def update_estimate_history(self):
        for i in range(self.environment.arm_count):
            self.estimate_history[i].append(self.estimated_rewards[i])

    # TODO:Remove redudancy in zeros
    def update_reward_history_per_arm(self, arm_selected, reward):
        for i in range(self.environment.arm_count):
            if arm_selected == i:
                self.reward_history_per_arm[i].append(reward)
            else:
                self.reward_history_per_arm[i].append(0)

    def update(self,):
        arm = self.select_arm()
        reward = self.environment.get_reward(arm)

        self.update_estimate(arm_selected=arm, reward=reward)

        self.actions_taken.append(arm)
        self.epochs_taken += 1
        # self.past_strategy_used

        self.optimal_action_history.append(
            (arm == self.environment.optimal_arm).item())

        self.reward_history.append(reward)
        self.update_reward_history_per_arm(arm_selected=arm, reward=reward)

        self.update_estimate_history()
        return reward

    def update_estimate(self, arm_selected, reward):
        # Copied from older version
        self.counts[arm_selected] += 1  # Track of each arm selected

        old_value = self.estimated_rewards[arm_selected]

        self.estimated_rewards[arm_selected] = old_value + (
            1/self.counts[arm_selected])*(reward - old_value)

    def initialize_optimistically(self, value: float, increment: bool = False):
        """
        Intialize estimated rewards

        Args : 
                value : intialize to specific value
                increment : intial estimate = base_truth +  'value' 
        """

        if increment:
            self.estimated_rewards = (self.environment.base_truth + value)
        else:
            self.estimated_rewards = np.ones(self.environment.arm_count)*value

    def initialize_zeros(self):
        """
        Initialize estimated rewards to 0

        """
        self.estimated_rewards = np.zeros(self.environment.arm_count)

    def reinitalize_agent(self):
        self.reward_history = []
