# Third-part libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


# Modules imported
from .environment import Environment
from .agent import Agent
from .strategy import base, Strategy, UCB, EpsilonGreedy, Exploit, Explore
from .graphs import put_label, plot_arm_estimate, plot_comparision, plot_cumulative_reward, plot_multiple_arms, plot_single_arm_history
from .utils import average_over_n_runs, compare_policies, run_experiment, save_image_to_disk, average_over_runs, compare_rewards


def main():
    # Random Seed #TODO: None for now
    random_seed = 88
    np.random.seed(random_seed)

    env = Environment(arm_count=10, max_mean=200, s_d=1)

    a = Agent(env, EpsilonGreedy(0.2))

    run_experiment(agent=a, steps=2000)

    plot_single_arm_history(a, arm_no=3)
    save_image_to_disk("single_arm_run")

    # agent_conf = [[env, EpsilonGreedy(0.4)], [env, EpsilonGreedy(0.2)], [
    #     env, UCB(2)]]

    # means = utils.compare_rewards(agent_conf=agent_conf, steps=500, runs=2000)

    # graph.compare_graphs(means, label=["Epsilon Greedy 0.4", "Epsilon Greedy 0.2", "UCB 2"], title="Cumultive reward over steps",
    #                      xlabel="Steps", ylabel="Total Reward")

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
