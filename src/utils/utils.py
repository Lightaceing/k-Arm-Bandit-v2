import numpy as np


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
