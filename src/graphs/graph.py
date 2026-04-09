import matplotlib.pyplot as plt
import numpy as np
import datetime

from utils.utils import save_image_to_disk


def put_label(ax, title, xlabel=None, ylabel=None):
    """
    Sets labels, titles, legend
    Args:
        title : title for graph
        xlabel : label for x axis
        ylabel : label for y axis
    """

    if xlabel is None:
        xlabel = "Steps"
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = "Rewards"
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.legend()


def plot_arm_estimate(agent, arm_no, max_steps=None, ax=None, custom_label=None, custom_title=None, save_image=False,  save_as="single_arm_estimate"):
    """
    Plot graph for one specific arm
    Args :
    agent : the agent whose arms are to be plotted
    arm_no : the arm to be plotted
    max_steps : the maximum no. of steps to be plotted
    ax : axis to plot the graph on | creates new if None is given
    custom_label : Custom label 
    save_image : Whether to save plot as an image or not | Def : False
    save_as : Name of the file | Def: single_arm_estimate 
    """

    if ax is None:
        fig, ax = plt.subplots()
    if max_steps is None:
        max_steps = agent.epochs_taken
    if custom_title is None:
        custom_title = f"Reward over steps for arm : {arm_no}"

    # Create label
    if custom_label == None:
        label = f"Arm {arm_no}"
    else:
        label = custom_label + f" Arm {arm_no}"

    # Create true_val np.array for plotting
    true_val = [agent.environment.base_truth[arm_no]] * \
        len(agent.estimate_history[arm_no])

    # Plot estimate
    ax.plot(agent.estimate_history[arm_no][:max_steps],
            label=label)
    # Plot true value

    ax.plot(true_val[:max_steps], linestyle="--",
            label=f"True Arm {arm_no}", alpha=0.6)

    put_label(ax, custom_title)
    if save_image is True:
        save_image_to_disk(plt, filename=save_as)


def plot_multiple_arms(agent, ax, only_top=False, top_count=3, max_steps=None, custom_label=None, custom_title=None, save_image=False,  save_as="multiple_arm_estimate"):
    """
    Plots estimated rewards of multiple arms of an agent
    Args :
    agent : the agent whose arms are to be plotted
    ax : axis to plot the graph on | creates new if None is given
    only_top : if only the top arms should be plotted | Def : All arms to be plotted
    top_count : No. of top estimated arms to be plotted | Def : 3 | 
    max_steps : the maximum no. of steps to be plotted
    custom_label : Custom label | Def : "Graph of multiple arms"
    save_image : Whether to save plot as an image or not | Def : False
    save_as : Name of the file | Def: multiple_arm_estimate
    """

    if ax == None:
        fig, ax = plt.subplots()
    if max_steps is None:
        max_steps = agent.epochs_taken

    incr_arms = np.argsort(agent.estimated_rewards)

    # Top arm and Bottom arm indexes
    top_arm_index = incr_arms[:top_count]
    bot_arm_index = incr_arms[top_count:]

    # Plot bottom arms
    if only_top == False:
        for arm in bot_arm_index:
            plot_arm_estimate(
                agent, arm_no=arm, max_steps=max_steps, ax=ax, custom_label=custom_label)
    # Plot top arms
    for arm in top_arm_index:
        plot_arm_estimate(
            agent, arm_no=arm, max_steps=max_steps, ax=ax, custom_label=custom_label)

    if custom_label is None:
        custom_title = "Graph of multiple arms"
    put_label(ax, title=custom_title)
    if save_image is True:
        save_image_to_disk(plt, filename=save_as)


def plot_single_arm_history(agent, arm_no, max_steps, ax=None, custom_title=None, save_image=False,  save_as="single_arm_history"):
    """
    Plot graph for one specific arm
    Args :
    agent : the agent whose arms are to be plotted
    arm_no : the arm to be plotted
    max_steps : the maximum no. of steps to be plotted
    ax : axis to plot the graph on | creates new if None is given
    custom_title : Custom title : 
    save_image : Whether to save plot as an image or not | Def : False
    save_as : Name of the file | Def: single_arm_estimate 
    """

    if ax == None:
        fig, ax = plt.subplots()
    if max_steps is None:
        max_steps = agent.epochs_taken

    true_val = [agent.environment.base_truth[arm_no]] * \
        len(agent.estimate_history[arm_no])

    ax.plot(agent.estimate_history[arm_no][:max_steps], label=f"Arm {arm_no}")

    ax.plot(true_val[:max_steps], linestyle="--",
            label=f"True Arm {arm_no}", alpha=0.6)

    if custom_title is None:
        custom_title = f"Reward over steps for arm : {arm_no}"
    put_label(ax, custom_title)

    if save_image is True:
        save_image_to_disk(plt, filename=save_as)


def plot_comparision(agents, only_top=False, top_count=3, max_steps=None, custom_title=None, save_image=False,  save_as="strategy_comparison"):
    """
    Plot graph for one specific arm
    Args :
    agents : list of agents who are being compared
    only_top : if only the top arms should be plotted | Def : All arms to be plotted
    top_count : No. of top estimated arms to be plotted | Def : 3 | 
    max_steps : the maximum no. of steps to be plotted
    custom_title : Custom title | Def : strategy1  VS  Strategy"
    save_image : Whether to save plot as an image or not | Def : False
    save_as : Name of the file | Def: strategy_comparison
    """

    fig, ax = plt.subplots()

    if max_steps is None:
        max_steps = agents[0].epochs_taken
    if custom_title is None:
        custom_title = f"{agents[0].strategy.name} VS {agents[1].strategy.name} Strategy"

    for agent in agents:
        plot_multiple_arms(agent, ax, only_top=only_top, top_count=top_count,
                           max_steps=max_steps, custom_label=agent.strategy.name, custom_title=custom_title)

    put_label(ax, title=custom_title)
    if save_image is True:
        save_image_to_disk(plt, filename=save_as)
