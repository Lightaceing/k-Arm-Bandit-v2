# Third-part libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


# Modules imported
from src import Environment, Agent

from .strategy import base, Strategy, UCB, EpsilonGreedy, Exploit, Explore
from .graphs import put_label, plot_arm_estimate, plot_comparision, plot_cumulative_reward, plot_multiple_arms, plot_single_arm_history
from .utils import average_over_n_runs, compare_policies, run_experiment, save_image_to_disk, average_over_runs, compare_rewards

new_environment = Environment(arm_count=5, max_mean=20)
new_agent = Agent(
    environment=new_environment,
    strategy=EpsilonGreedy(epsilon_value=0.3)
)

average_over_n_runs(agent=new_agent, steps=500, n_runs=100)


# run_experiment(
#     agent=new_agent, steps=1000)

plot_multiple_arms(agent=new_agent, ax=None,  only_top=True)

plt.show()
plt.close()
