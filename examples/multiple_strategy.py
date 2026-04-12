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

# Created Enviroment
env = environment.Environment(arm_count=10, max_mean=200, s_d=1)

# Created Agent and config
agent_conf = [[env, strategy.EpsilonGreedy(0.4)], [env, strategy.EpsilonGreedy(0.2)], [
    env, strategy.UCB(2)]]

# Run experiment by comparing rewards b/w multiple agents averaging over 2000 runs
# This creates multiple agents based on config and runs them on it own.
means = util.compare_rewards(agent_conf=agent_conf, steps=500, runs=2000)

# Plot all of their graphs against each other
graph.compare_graphs(means, label=["Epsilon Greedy 0.4", "Epsilon Greedy 0.2", "UCB 2"], title="Cumultive reward over steps",
                     xlabel="Steps", ylabel="Total Reward")


# To see the arms
plt.show()


# Save the images
# util.save_image_to_disk("single_arm_run")
