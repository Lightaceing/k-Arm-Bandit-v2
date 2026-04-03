import numpy as np

# Implementing Class for entire learning algo


class Agent:

    def __init__(self, arm_count=10, strategy="epsilon", value=1):

        # Arm count should be +ve
        if arm_count <= 0:
            raise ValueError("Enter a value greater than 0.")
        if type(arm_count) != int:
            raise TypeError("Only integer values allowed")

        self.arm_count = arm_count
        self.estimated_rewards = np.zeros(
            shape=arm_count)  # Default is all zeros
        self.counts = np.zeros(shape=arm_count)
        self.strategy = strategy
        self.value = value

        # For graph
        self.history = []
        self.arm_select_history = []
        # Creating history
        for i in range(0, self.arm_count):
            self.history.append([0])

    def initialize_empty_reward(self):
        """
        Initialize all rewards as 0
        """
        self.estimated_rewards = np.zeros(
            shape=self.arm_count)

    def update_estimate(self, arm_selected, reward):
        """
        Updates estimates using incremental update
        Tracks estimate history
        """
        self.arm_select_history.append(int(arm_selected))

        self.counts[arm_selected] += 1  # Track of each arm selected

        old_value = self.estimated_rewards[arm_selected]
        self.estimated_rewards[arm_selected] = old_value + (
            1/self.counts[arm_selected])*(reward - old_value)

        # Update history
        for i in range(0, self.arm_count):
            self.history[i].append(
                self.estimated_rewards[i])

    def epsilon_greedy(self, epsilon=0.2):
        """
        Explores with a probability of epsilon
        Exploits with a probability of 1-epsilon
        """
        if not isinstance(epsilon, (int, float)):
            raise TypeError("Accepts only integer and float values")

        val = np.random.uniform(0, 1)
        if (val <= epsilon):
            arm = self.explore()
        else:
            arm = self.exploit()

        return arm

    def optimistic_init(self, value):
        self.estimated_rewards = np.ones(shape=self.arm_count)*value

    def exploit(self):
        """
        Selects the arm with the largest estimated value
        """
        arm = np.argmax(self.estimated_rewards)
        return arm

    def explore(self):
        """
        Selects the arm randomly
        """
        random_arm = np.random.randint(0, self.arm_count)
        return random_arm

    def get_history(self):
        return self.history

    def update_histroy(self, arm_selected):
        # self.history = self.history.append(self.estimated_rewards)
        # self.arm_select_history = self.arm_select_history.append(arm_selected)
        pass

    def ucb(self, c=2):
        for i in range(len(self.counts)):
            if self.counts[i] == 0:
                return i
        total_count = np.sum(self.counts)

        explore_criteria = c * \
            np.sqrt(np.log(total_count)/(np.array(self.counts)))
        arm = np.argmax(self.estimated_rewards + explore_criteria)
        return arm
