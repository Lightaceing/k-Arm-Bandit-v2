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


def update(agent, env, technique, value=0.2):

    tech_config = ["epsilon", "exploit", "explore", "optimistic_init"]
    if technique not in tech_config:
        raise ValueError("The value should be one of ", tech_config)

    if technique == "epsilon":
        arm_selected = agent.epsilon_greedy(value)
    elif technique == "exploit":
        arm_selected = agent.exploit()
    elif technique == "explore":
        arm_selected = agent.explore()
    elif technique == "optimistic_init":
        arm_selected = agent.exploit()

    reward = pull_arm(environment=env, arm_selected=arm_selected, s_d=1)
    agent.update_estimate(arm_selected, reward)


def run_experiment(Agent, steps, technique, value, env_config):

    # Create Environment
    env = create_environment(
        arm_count=env_config["arm_count"], max_mean=env_config["max_mean"])

    # Create Agent
    agent = Agent(arm_count=env_config["arm_count"])

    # Init for optimistic init
    if technique == "optimistic_init":
        agent.optimistic_init()

    agent.optimistic_init()

    for _ in range(steps):
        update(agent, env, technique, value)

    return agent, env
