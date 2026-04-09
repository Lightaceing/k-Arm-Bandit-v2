import numpy as np
import matplotlib.pyplot as plt
from envrionment.environment import Environment
from agent.agents import Agent
from strategy import Strategy, Exploit, Explore, EpsilonGreedy, UCB

from graphs.graph import plot_single_arm_history, plot_multiple_arms, plot_arm_estimate, plot_comparision
from utils.utils import average_over_n_runs, compare_policies, run_experiment

from logger_utils.logger import *

# Random Seed #TODO: None for now
random_seed = None
np.random.seed(random_seed)


env = Environment(arm_count=10, max_mean=200, s_d=30)

agent1 = Agent(env, strategy=EpsilonGreedy(0.4))
agent2 = Agent(env, strategy=UCB(2))

agents = [agent1, agent2]

compare_policies(agents=agents, steps=2000)
plot_comparision(agents, True, 2, save_as="comparison between two strategies")
plt.show()
plt.close()
