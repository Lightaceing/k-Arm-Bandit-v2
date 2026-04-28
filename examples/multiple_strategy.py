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
agent1 = agents.Agent(environment=env, strategy=strategy.EpsilonGreedy(0.3))
agent2 = agents.Agent(environment=env, strategy=strategy.UCB(2))

all_agents = [agent1, agent2]

# Run all the agents
util.compare_policies(agent_list=all_agents, steps=1000)

# Plot all of their graphs against each other
graph.compare_reward_history(all_agents, title="Cumultive reward over steps",
                             xlabel="Steps", ylabel="Total Reward", plot=True)

graph.compare_optimal_action(all_agents, title="Cumultive reward over steps",
                             xlabel="Steps", ylabel="Total Reward", plot=True)


# Save the images
# util.save_image_to_disk("single_arm_run")
