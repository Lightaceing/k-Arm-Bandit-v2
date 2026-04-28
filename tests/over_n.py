# Third-part libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Core
import src.environment.environment as environment
import src.agent.agents as agents

# Strategy
import src.strategy as strategy

# Utilities
import src.graphs.graph as graph
import src.utils.util as util

# Logging
import src.logger_utils.runtime_logs as runtime_logs


random_seed = None
np.random.seed(random_seed)

# ========================Code Start==========#

# Create Enviroment
env = environment.Environment(arm_count=10, max_mean=200, s_d=1)

# Create Agent
agent = agents.Agent(environment=env, strategy=strategy.EpsilonGreedy(0.4))

# util.run_experiment(agent=agent, steps=600)
average_reward = util.average_over_n_runs(
    agent=agent, steps=600, n_runs=200)


graph.plot_cumulative_reward(mean=average_reward,   label="MEOW")

# print("average_reward", average_reward.shape)
# print("average_history", average_history.shape)


plt.show()
