import numpy as np
from agent import Agent

from utils.utils import *
from graphs.graph import *
from logger_utils.logger import *

# Random Seed #TODO: None for now
random_seed = None
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

agents, env = compare_policies(Agent,
                               steps=experiment_config["steps"], strategy=experiment_config["strategy"], value=0.2, env_config=env_config)


# Graph printed
compare_two_graphs(agents, env, True, max_steps=200, top_count=2)
