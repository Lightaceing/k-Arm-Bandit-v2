import matplotlib.pyplot as plt
import numpy as np
import datetime


def single_plot(agent, arm_no, max_steps, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(agent.estimate_history[arm_no][:max_steps], label=f"Arm {arm_no}")
    true_val = [agent.environment.base_truth[arm_no]] * \
        len(agent.estimate_history[arm_no])
    ax.plot(true_val[:max_steps], linestyle="--",
            label=f"True Arm {arm_no}", alpha=0.6)


def plot_single_arm_history(agent, arm_no, max_steps, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(agent.estimate_history[arm_no][:max_steps], label=f"Arm {arm_no}")
    true_val = [agent.environment.base_truth[arm_no]] * \
        len(agent.estimate_history[arm_no])
    ax.plot(true_val[:max_steps], linestyle="--",
            label=f"True Arm {arm_no}", alpha=0.6)
    ax.set_ylabel("Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Reward over steps for arm : {arm_no}")
    ax.legend()
    plt.show()


def plot_estimates(agent, only_top=False, top_count=3, max_steps=520, custom_title=None):
    fig, ax = plt.subplots()
    if only_top:
        top_arms = np.argsort(agent.estimated_rewards)[-1*top_count:][::-1]
        for arm in top_arms:
            single_plot(
                agent, arm_no=arm, max_steps=max_steps, ax=ax)
    else:
        for i in range(agent.environment.arm_count):
            single_plot(
                agent, arm_no=i, max_steps=max_steps, ax=ax)

    ax.set_ylabel("Reward on each step")
    ax.set_xlabel("Steps")
    ax.set_title(f"Reward over steps for arm : {agent.environment.arm_count}")
    ax.legend()
    plt.show()


def save_graph(graph):  # TODO:Improve
    path = "../results/"
    filename = path + "record_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
    plt.savefig(filename, )
    plt.close()
    # Log plot saved
