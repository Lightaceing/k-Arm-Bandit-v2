import datetime
import csv


CSV_FILE = "results/records_" + \
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv"


def log_run(agent, avg_reward, total_regret):
    with open("runs.csv", "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            datetime.now(),
            type(agent.strategy).__name__,
            agent.environment.arm_count,
            agent.steps,
            getattr(agent.strategy, "epsilon", None),
            getattr(agent.strategy, "c", None),
            avg_reward,
            total_regret
        ])


def init_csv():
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "strategy", "arm",
                        "current reward", "cumulative reward", "optimal"])


def record_csv(epoch, strategy, arm, reward, cumulative, optimal):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, strategy, arm, reward, cumulative, optimal])
