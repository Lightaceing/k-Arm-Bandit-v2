import datetime


def take_logs(agent):
    path = "../results/"
    filename = path + "record_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".txt"
    with open(filename, "w") as f:
        f.write(str(agent.arm_select_history))
        f.write("")
        f.write(str(agent.history))
