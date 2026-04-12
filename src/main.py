# Third-part libraries
from src import sys, np, plt

# Modules imported
from src import environment, agents, strategy, graph, util


def main():
    # Random Seed #TODO: None for now
    random_seed = None
    np.random.seed(random_seed)

    env = environment.Environment(arm_count=10, max_mean=200, s_d=1)

    a = agents.Agent(env, strategy.EpsilonGreedy(0.2))

    util.run_experiment(agent=a, steps=2000)

    graph.plot_single_arm_history(a, arm_no=3)
    util.save_image_to_disk("single_arm_run")

    # agent_conf = [[env, EpsilonGreedy(0.4)], [env, EpsilonGreedy(0.2)], [
    #     env, UCB(2)]]

    # means = utils.compare_rewards(agent_conf=agent_conf, steps=500, runs=2000)

    # graph.compare_graphs(means, label=["Epsilon Greedy 0.4", "Epsilon Greedy 0.2", "UCB 2"], title="Cumultive reward over steps",
    #                      xlabel="Steps", ylabel="Total Reward")

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
