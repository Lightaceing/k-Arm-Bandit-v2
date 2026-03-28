import numpy as np
from agent import LearnAlgo

from enviroment import create_enviroment
from utils import pull_arm

random_seed = None
np.random.seed(random_seed)


def update():
    arm_selected = l1.epsilon_greedy()
    reward = pull_arm(enviroment=env, arm_selected=arm_selected, s_d=1)
    l1.update_estimate(arm_selected, reward)


arm_count = 10

# ==========Main Loop=========#

# Create Enviroment

env = create_enviroment(arm_count=arm_count, max_mean=20)
print("Enviroment value ", env)


l1 = LearnAlgo()


for i in range(1, 1000):
    update()

# ===============Ending Loop========#


print("*"*20)

print(l1.estimated_rewards)


gg = np.zeros(shape=(arm_count, 2))

print(gg)