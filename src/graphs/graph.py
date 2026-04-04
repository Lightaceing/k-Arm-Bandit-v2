import matplotlib.pyplot as plt
import numpy as np
import datetime


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
    path = "../results/"
    filename = path + "record_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


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
    path = "../results/"
    filename = path + "record_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def compare_true_value_with_estimate(agent, env, only_top=False, top_count=3, max_steps=520):
    """
    Plot true values and estimated values over time

    """
    if max_steps == None:
        max_steps = int(np.sum(agent.counts))
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
            ax.plot(agent.history[arm][:max_steps], label=f"Arm {arm}")
            true_val = [env[arm]]*len(agent.history[arm])
            ax.plot(true_val[:max_steps], linestyle="--",
                    label=f"True Arm {arm}", alpha=0.6)
    else:
        for i in range(0, agent.arm_count):
            ax.plot(agent.history[i][:max_steps], label=f"Arm {i}")
            true_val = [env[i]]*len(agent.history[i])
            ax.plot(true_val[:max_steps], linestyle="--",
                    label=f"True Arm {i}", alpha=0.6)

    ax.set_ylabel("Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Reward over steps for all arms")
    ax.legend()
    path = "../results/"
    filename = path + "record_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def compare_true_value_with_estimate_steps(agent, env, max_steps, only_top=False, top_count=3, legend=True):
    """
    Plot true values and estimated values over time

    """
    if max_steps == None:
        max_steps = int(np.sum(agent.counts))
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
            ax.plot(agent.history[arm][:max_steps], label=f"Arm {arm}")
            true_val = [env[arm]]*len(agent.history[arm])
            ax.plot(true_val[:max_steps], linestyle="--",
                    label=f"True Arm {arm}", alpha=0.6)
    else:
        for i in range(0, agent.arm_count):
            ax.plot(agent.history[i][:max_steps], label=f"Arm {i}")
            true_val = [env[i]]*len(agent.history[i])
            ax.plot(true_val[:max_steps], linestyle="--",
                    label=f"True Arm {i}", alpha=0.6)

    ax.set_ylabel("Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Reward over steps for all arms")
    if legend:
        ax.legend()
    plt.show()


def save_graph(graph):
    path = "../results/"
    filename = path + "record_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
    plt.savefig(filename, )
    plt.close()
    # Log plot saved


def compare_two_graphs(agent, env, only_top=False, top_count=3, max_steps=520):
    """
    Plot true values and estimated values over time

    """
    if max_steps == None:
        max_steps = int(np.sum(agent.counts))
    if top_count < 0:
        raise ValueError("Top count cant be negative")
    if type(top_count) != int:
        raise TypeError("Top count should be an integer")
    if type(only_top) != bool:
        raise TypeError("Takes a boolean")

    fig, ax = plt.subplots()

    if only_top:
        for each_agent in agent:
            # Only prints the top 3 arms
            top_arms = np.argsort(
                each_agent.estimated_rewards)[-1*top_count:][::-1]
            for arm in top_arms:
                ax.plot(each_agent.history[arm]
                        [:max_steps], label=f"Arm with strategy {each_agent.strategy} {arm}")
                true_val = [env[arm]]*len(each_agent.history[arm])
                ax.plot(true_val[:max_steps], linestyle="--",
                        label=f"True Arm {arm}", alpha=0.6)
    else:
        for each_agent in agent:
            for i in range(0, each_agent.arm_count):
                ax.plot(each_agent.history[i][:max_steps], label=f"Arm {i}")
                true_val = [env[i]]*len(each_agent.history[i])
                ax.plot(true_val[:max_steps], linestyle="--",
                        label=f"True Arm {i}", alpha=0.6)

    ax.set_ylabel("Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Reward over steps for all arms")
    ax.legend()
    path = "../results/"
    filename = path + "record_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
