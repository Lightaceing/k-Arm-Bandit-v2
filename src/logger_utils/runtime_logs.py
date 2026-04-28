import datetime


# LOG File generated once per script run
LOG_FILE = "logs/" + "record_" + \
    datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".txt"


def take_logs(agent):
    filename = "../results/" + "record_" + \
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".txt"
    with open(filename, "w") as f:
        f.write(str(agent.arm_select_history))
        f.write("")
        f.write(str(agent.history))


def record_logs(log: str, filename=None):
    # Write log
    with open(LOG_FILE, "a") as f:
        f.write(log)
        f.write("\n")

    return filename
