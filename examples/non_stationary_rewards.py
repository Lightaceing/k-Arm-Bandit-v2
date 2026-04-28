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

# ========================Code Start==========#


# Create Enviroment
env = environment.Environment(arm_count=10, max_mean=200, s_d=1)

# Create Agent
agent = agents.Agent(environment=env, strategy=strategy.EpsilonGreedy(0.7))

# Run Experiment
util.run_experiment(agent=agent, steps=600)

# New enviroment creaated
new_env = environment.Environment(arm_count=10, max_mean=1000, s_d=1)

# Updating environment
agent.update_enviroment(new_env)

util.run_experiment(agent=agent, steps=600)

graph.plot_multiple_arm_estimate(agent, only_top=True, plot=True)
