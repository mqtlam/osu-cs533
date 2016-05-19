import numpy as np

from mdp import MDP
from parking_mdp import ParkingMDP, ParkingAction
from policy import RandomParkingPolicy, SafeRandomParkingPolicy, SafeNoHandicapRandomParkingPolicy
from simulator import MDPSimulator
from qlearner import QLearner

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


### PART II: MDP 1
num_rows = 10
mdp1 = ParkingMDP(num_rows, handicap_reward=-100)

# create simulator
initial_state1 = mdp1.get_state_id(0, num_rows-1, 0, 0)
simulator = MDPSimulator(mdp1, initial_state=initial_state1)

# run simulation 1: random policy
policy = RandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward

# run simulation 2: safer random policy
policy = SafeRandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward

# run simulation 3: safer no handicap random policy
policy = SafeNoHandicapRandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward


### PART II: MDP 2
mdp2 = ParkingMDP(num_rows, handicap_reward=100)

# create simulator
initial_state2 = mdp2.get_state_id(0, num_rows-1, 0, 0)
simulator = MDPSimulator(mdp2, initial_state=initial_state2)

# run simulation 1: random policy
policy = RandomParkingPolicy(mdp2, park_probability=0.1)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward

# run simulation 2: safer random policy
policy = SafeRandomParkingPolicy(mdp2, park_probability=0.1)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward

# run simulation 3: safer no handicap random policy
policy = SafeNoHandicapRandomParkingPolicy(mdp2, park_probability=0.1, handicap_probability=1.0)
avg_reward = 0
num_trials = 10000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print avg_reward


### PART III
policy = RandomParkingPolicy(mdp1, park_probability=0.1)
qlearner = QLearner(mdp1, initial_state1)

num_learning_trials = 10000
num_simulation_trials = 1000
num_learning_epochs = 5
for epoch in range(num_learning_epochs):
    print "Running learning {}...".format(epoch)
    # run learning trials
    for trial in range(num_learning_trials):
        qlearner.run_learning_trial()

    # run simulation trials
    print "Running simulation {}...".format(epoch)
    avg_reward = 0
    for trial in range(num_simulation_trials):
        (total_reward, state_seq, action_seq) = qlearner.run_simulation_trial()
        avg_reward += total_reward
    avg_reward = 1.*avg_reward/num_simulation_trials
    print avg_reward
