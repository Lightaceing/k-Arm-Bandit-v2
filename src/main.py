import numpy as np
from agent import Agent

from utils.utils import *
from graphs.graph import *
from logger_utils.logger import *

# Random Seed #TODO: None for now
random_seed = 44
np.random.seed(random_seed)

# Just for Refrence
# tech_config = ["epsilon", "exploit", "explore", "ucb"]


env_config = {"arm_count": 10,
              "max_mean": 20}

experiment_config = {
    "steps": 1000,
    "strategy": ["epsilon", "ucb", "exploit"]
}

# ==========Main Loop=========#

# Experiment Ran


agents, env, avg_rewards = compare_over_n_runs(Agent,
                                               steps=experiment_config["steps"], strategy="epsilon", value=0.07, env_config=env_config, n_runs=20000)

# Graph

# compare_true_value_with_estimate(
#     agents, env, only_top=True, top_count=3, max_steps=220, custom_title="Estimates over 220 runs")


plt.plot(avg_rewards)
plt.title("Average Reward over Time")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.show()
