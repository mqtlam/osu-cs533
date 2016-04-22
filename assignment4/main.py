from bandit import SBRDBandit

### PROBLEM 2

arm_params = [(1, 0.1)]
for a in range(9):
    arm_params.append((0.05, 1))
bandit1 = SBRDBandit(arm_params)

arm_params = []
for a in range(20):
    arm_params.append((1.*(a+1)/20, 0.1))
bandit2 = SBRDBandit(arm_params)
