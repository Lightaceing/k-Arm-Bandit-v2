import datetime
import csv


CSV_FILE = "results/records_" + \
    datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"


def init_csv():
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "strategy", "arm",
                        "current reward", "cumulative reward", "optimal"])


def record_csv(epoch, strategy, arm, reward, cumulative, optimal):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, strategy, arm, reward, cumulative, optimal])
