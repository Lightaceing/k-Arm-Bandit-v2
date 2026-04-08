import numpy as np
import matplotlib.pyplot as plt
from envrionment.environment import Environment
from agent.agents import Agent
from strategy import Strategy, Exploit, Explore, EpsilonGreedy, UCB

from graphs.graph import plot_single_arm_history, plot_estimates
from utils.utils import average_over_n_runs

from logger_utils.logger import *

# Random Seed #TODO: None for now
random_seed = None
np.random.seed(random_seed)


env = Environment(arm_count=10, max_mean=200, s_d=30)

agent1 = Agent(env, strategy=EpsilonGreedy(0.3))


for _ in range(50000):
    agent1.update()

plot_estimates(agent1, True, 3)
