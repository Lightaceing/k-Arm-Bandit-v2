import matplotlib.pyplot as plt
import numpy as np


def plot_single_arm_history(agent, arm_no=0):
    """
    Plot a single arm's reward over multiple steps

    """
    if arm_no < 0:
        raise ValueError("Arm cant be negative")
    if type(arm_no) != int:
        raise TypeError("Arm no should be an integer")
    fig, ax = plt.subplots()

    ax.plot(agent.history[arm_no])

    ax.set_ylabel("Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Reward over steps for arm : {arm_no}")
    ax.legend()

    plt.show()


def plot_multi_arm_history(agent, only_top=False, top_count=3):
    """
    Plot all the arm's reward over multiple steps

    """

    if top_count < 0:
        raise ValueError("Top count cant be negative")
    if type(top_count) != int:
        raise TypeError("Top count should be an integer")
    if only_top != bool:
        raise TypeError("Takes a boolean")

    fig, ax = plt.subplots()

    if only_top:
        # Only prints the top 3 arms
        top_arms = np.argsort(
            np.array(agent.estimated_rewards))[-1*top_count:][::-1]
        for arm in top_arms:
            ax.plot(agent.history[arm], label=f"Arm {arm}")
    else:
        for i in range(0, agent.arm_count):
            ax.plot(agent.history[i], label=f"Arm {i}")

    ax.set_ylabel("Estimated Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Estimated Reward over steps for all arms")
    ax.legend()
    plt.show()


def compare_true_value_with_estimate(agent, env, only_top=False, top_count=3):
    """
    Plot true values and estimated values over time

    """

    if top_count < 0:
        raise ValueError("Top count cant be negative")
    if type(top_count) != int:
        raise TypeError("Top count should be an integer")
    if type(only_top) != bool:
        raise TypeError("Takes a boolean")

    fig, ax = plt.subplots()

    if only_top:
        # Only prints the top 3 arms
        top_arms = np.argsort(
            agent.estimated_rewards)[-1*top_count:][::-1]
        for arm in top_arms:
            ax.plot(agent.history[arm], label=f"Arm {arm}")
            true_val = [env[arm]]*len(agent.history[arm])
            ax.plot(true_val, linestyle="--",
                    label=f"True Arm {arm}", alpha=0.6)
    else:
        for i in range(0, agent.arm_count):
            ax.plot(agent.history[i], label=f"Arm {i}")
            true_val = [env[i]]*len(agent.history[i])
            ax.plot(true_val, linestyle="--", label=f"True Arm {i}", alpha=0.6)

    ax.set_ylabel("Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Reward over steps for all arms")
    ax.legend()
    plt.show()
