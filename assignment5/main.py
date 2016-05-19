import numpy as np

from mdp import MDP
from parking_mdp import ParkingMDP, ParkingAction
from policy import Policy, RandomParkingPolicy, SafeRandomParkingPolicy, SafeNoHandicapRandomParkingPolicy
from simulator import MDPSimulator
from qlearner import QLearner

from mdp_optimization import InfiniteHorizonPolicyOptimization

### SET UP
num_rows = 10
discount_factor = 0.99

# MDP 1
mdp1 = ParkingMDP(num_rows, handicap_reward=-100)
mdp1.save_to_file("MDP_parking1.txt")
initial_state1 = mdp1.get_state_id(0, num_rows-1, 0, 0)

# MDP 2
mdp2 = ParkingMDP(num_rows, handicap_reward=100)
mdp2.save_to_file("MDP_parking2.txt")
initial_state2 = mdp2.get_state_id(0, num_rows-1, 0, 0)


### PART II: MDP 1

# create simulator
simulator = MDPSimulator(mdp1, initial_state=initial_state1)

# run simulation 1: random policy
policy = RandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 random policy: {0}".format(avg_reward)

# run simulation 2: safer random policy
policy = SafeRandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 safer random policy: {0}".format(avg_reward)

# run simulation 3: safer no handicap random policy
policy = SafeNoHandicapRandomParkingPolicy(mdp1, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 safer no handicap random policy: {0}".format(avg_reward)

# run simulation 4: optimal policy
_, optimal_policy = InfiniteHorizonPolicyOptimization.policy_iteration(mdp1, discount_factor)
policy = Policy(optimal_policy)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state1)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP1 optimal policy: {0}".format(avg_reward)
print


### PART II: MDP 2

# create simulator
simulator = MDPSimulator(mdp2, initial_state=initial_state2)

# run simulation 1: random policy
policy = RandomParkingPolicy(mdp2, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 random policy: {0}".format(avg_reward)

# run simulation 2: safer random policy
policy = SafeRandomParkingPolicy(mdp2, park_probability=0.1)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 safer random policy: {0}".format(avg_reward)

# run simulation 3: safer no handicap random policy
policy = SafeNoHandicapRandomParkingPolicy(mdp2, park_probability=0.1, handicap_probability=1.0)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 safer handicap random policy: {0}".format(avg_reward)

# run simulation 4: optimal policy
_, optimal_policy = InfiniteHorizonPolicyOptimization.policy_iteration(mdp2, discount_factor)
policy = Policy(optimal_policy)
avg_reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator.reset(initial_state=initial_state2)
    (total_reward, state_seq, action_seq) = simulator.run_simulation(policy)
    avg_reward += total_reward
avg_reward = 1.*avg_reward/num_trials
print "MDP2 optimal policy: {0}".format(avg_reward)
print


### PART III: MDP 1
qlearner = QLearner(mdp1, initial_state1)

num_learning_trials = 10000
num_simulation_trials = 1000
num_learning_epochs = 20
for epoch in range(num_learning_epochs):
    for trial in range(num_learning_trials):
        qlearner.run_learning_trial()

    avg_reward = 0
    for trial in range(num_simulation_trials):
        (total_reward, state_seq, action_seq) = qlearner.run_simulation_trial()
        avg_reward += total_reward
    avg_reward = 1.*avg_reward/num_simulation_trials
    print "MDP1 epoch {0}: {1}".format(epoch, avg_reward)
print


### PART III: MDP 2
qlearner = QLearner(mdp2, initial_state2)

num_learning_trials = 10000
num_simulation_trials = 1000
num_learning_epochs = 20
for epoch in range(num_learning_epochs):
    for trial in range(num_learning_trials):
        qlearner.run_learning_trial()

    avg_reward = 0
    for trial in range(num_simulation_trials):
        (total_reward, state_seq, action_seq) = qlearner.run_simulation_trial()
        avg_reward += total_reward
    avg_reward = 1.*avg_reward/num_simulation_trials
    print "MDP2 epoch {0}: {1}".format(epoch, avg_reward)
