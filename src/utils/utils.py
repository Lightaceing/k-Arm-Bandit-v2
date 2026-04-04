import numpy as np
from enviroment import create_environment
from agent import Agent


def pull_arm(environment, arm_selected, s_d=1):
    """
    Selects an arm based on the tech. given by select_arm fn. and generates a reward based on the arm's mean selected

    """
    arm_mean = environment[arm_selected]
    reward = np.random.normal(loc=arm_mean, scale=s_d)

    return reward


def update(agent, env):

    tech_config = ["epsilon", "exploit", "explore", "optimistic_init", "ucb"]
    if agent.strategy not in tech_config:
        raise ValueError("The value should be one of ", tech_config)
    arm_selected = 0

    if agent.strategy == "epsilon":
        arm_selected = agent.epsilon_greedy(
            agent.value)
    elif agent.strategy == "exploit":
        arm_selected = agent.exploit()
    elif agent.strategy == "explore":
        arm_selected = agent.explore()
    elif agent.strategy == "optimistic_init":
        arm_selected = agent.exploit()
    elif agent.strategy == "ucb":
        arm_selected = agent.ucb(c=agent.value)

    reward = pull_arm(environment=env, arm_selected=arm_selected, s_d=1)
    agent.update_estimate(arm_selected, reward)

    agent.update_histroy(arm_selected)


def run_experiment(Agent, steps, strategy, value, env_config):

    # Create Environment
    env = create_environment(
        arm_count=env_config["arm_count"], max_mean=env_config["max_mean"])

    # Create Agent
    agent = Agent(arm_count=env_config["arm_count"],
                  value=value, strategy=strategy)

    # Run Epochs
    for _ in range(steps):
        update(agent, env)

    return agent, env


def compare_policies(Agent, steps, strategy, value, env_config):
    # Create Environment
    env = create_environment(
        arm_count=env_config["arm_count"], max_mean=env_config["max_mean"])

    agents = []
    for each_strategy in strategy:
        # Agent created
        agent = Agent(arm_count=env_config["arm_count"],
                      value=value, strategy=each_strategy)

        # Run Epochs for agent1
        for _ in range(steps):
            update(agent, env)

        agents.append(agent)

    return agents, env
