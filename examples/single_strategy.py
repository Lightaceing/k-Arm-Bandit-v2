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
agent = agents.Agent(environment=env, strategy=strategy.EpsilonGreedy(0.2))

# Run Experiment
util.run_experiment(agent=agent, steps=2000)

# Plot the graph of a specific arm
graph.plot_single_arm_history(agent=agent, arm_no=3)


# Plot the graph of a top 4 arm estimates
graph.plot_multiple_arms(agent=agent, ax=None, only_top=True, top_count=4)

# To see the arms
plt.show()


# Save the images
# util.save_image_to_disk("single_arm_run")
