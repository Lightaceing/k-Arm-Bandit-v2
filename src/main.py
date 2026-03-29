import numpy as np
from agent import Agent

from utils import run_experiment
from graph import plot_single_arm_history, plot_multi_arm_history, compare_true_value_with_estimate

# Random Seed #TODO: None for now
random_seed = None
np.random.seed(random_seed)

# Just for Refrence
# tech_config = ["epsilon", "exploit", "explore"]


env_config = {"arm_count": 10,
              "max_mean": 20}

experiment_config = {
    "steps": 1000,
    "technique": "epsilon"
}

# ==========Main Loop=========#

# Experiment Ran
agent, env = run_experiment(Agent,
                            steps=experiment_config["steps"], technique=experiment_config["technique"], value=0.3, env_config=env_config)

# ===============Ending Loop========#

# Printing graphs
compare_true_value_with_estimate(agent, env, True)
