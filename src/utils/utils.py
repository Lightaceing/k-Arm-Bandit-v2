import numpy as np
import os
import matplotlib.pyplot as plt
import functools


def average_over_n_runs(agent, steps, n_runs):

    all_rewards = []
    all_optimal = []
    all_estimate_history = []
    for _ in range(n_runs):
        # Run Epochs
        for _ in range(steps):
            agent.update()

        all_rewards.append(agent.reward_history)
        all_optimal.append(agent.optimal_action_history)
        all_estimate_history.append(agent.estimate_history)

        agent.initialize_zeros()
        agent.reinitalize_agent()
        # Creating history
        agent.estimate_history = []
        for _ in range(agent.environment.arm_count):
            agent.estimate_history.append([])
    average_reward = np.mean(np.array(all_rewards), axis=0)
    average_history = np.mean(all_estimate_history, axis=0)
    return average_reward, average_history


def compare_policies(agents, steps):
    for agent in agents:
        for _ in range(steps):
            agent.update()


def run_experiment(agent, steps):
    for _ in range(steps):
        agent.update()


def save_image_to_disk(plt, filename):
    if filename:
        if not filename.endswith('.png'):
            filename += '.png'

        os.makedirs("../results/", exist_ok=True)

        save_path = "../results/" + filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
