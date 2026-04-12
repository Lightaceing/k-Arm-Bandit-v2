import numpy as np
from src.logger_utils.runtime_logs import record_logs
import matplotlib.pyplot as plt
from src.agent.agents import Agent


def average_over_n_runs(agent: Agent, steps: int, n_runs: int):
    """
    Averages entire estimate history over run
    Args:
        agent : the agent to be trained.
        steps : no. of steps the agent is to run.
        n_runs : no. of runs over which is to be averaged.
    """

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


def compare_policies(agents: Agent, steps: int):
    """
    Returns multiple agents run 
    Args :
        agents : list of agents
        steps : no. of steps the agents is to be run.
    """
    for agent in agents:
        run_experiment(agent, steps)
        record_logs(
            f"Run for agent with strategy : {agent.strategy.name} completed.\n")


def run_experiment(agent: Agent, steps: int):
    """
    Trains agent over multiple steps
    Args :
        agent : the agent to be trained
        steps : no. of steps the agent is to be trained.

    """
    for _ in range(steps):
        agent.update()
    record_logs("Experiment Run Completed.\n")


def save_image_to_disk(filename: str):
    """
    Saves the plot
    Arg : 
        filename : name of the file
    """

    if filename:
        if not filename.endswith('.png'):
            filename += '.png'

        save_path = "results/" + filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        record_logs(f"Image Saved : {save_path}.")


def average_over_runs(agent_conf: list, steps: int, runs: int):
    """
    Averages reward gained over multiple runs for multiple agents
    Args : 
    agent_conf : agent configuration | [agent_environment, agent_strategy]
    steps : no. of steps the agents is to be run.
    runs : no. of runs over which rewards is to be averaged.
    """

    all_runs = []
    for i in range(runs):
        agent = Agent(agent_conf[0], agent_conf[1])
        run_experiment(agent, steps)
        all_runs.append(agent.reward_history)
    mean = np.mean(np.array(all_runs), axis=0)
    return mean


def compare_rewards(agent_conf: list, steps: int, runs: int):
    """
    Compare rewards between multiple agents
    Args : 
        agent_conf : agent configuration | [agent_environment, agent_strategy]
        steps : no. of steps the agents is to be run.
        runs : no. of runs over which rewards is to be averaged.
    """
    all_mean = []
    for agents in agent_conf:
        mean = average_over_runs(agents, steps, runs)
        all_mean.append(mean)

    return all_mean
