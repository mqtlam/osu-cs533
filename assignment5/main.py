import numpy as np

from mdp import MDP
from parking_mdp import ParkingMDP, ParkingAction
from policy import RandomParkingPolicy, SafeRandomParkingPolicy
from simulator import MDPSimulator

### helper functions to print stuff

def print_state_helper(state):
    if state == (-1, -1, -1, -1): # terminal state
        return "(terminal state)\t"
    else:
        a_or_b = "A" if state[0] == 0 else "B"
        row = str(state[1])
        occupied = "occupied" if state[2] == 1 else "unoccupied"
        parked = "parked" if state[3] == 1 else "unparked"
        state_string = "({}, {}, {}, {})".format(a_or_b, row, occupied, parked)
        return state_string

### PART II

# create MDP
mdp = ParkingMDP(10, handicap_reward=-100, collision_reward=-10000)

# create simulator
initial_state = 0
simulator = MDPSimulator(mdp, initial_state=initial_state)

# run simulation 1: random policy
policy = RandomParkingPolicy(mdp, park_probability=0.1)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward

# run simulation 2: safer random policy
policy = SafeRandomParkingPolicy(mdp, park_probability=0.1)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward
