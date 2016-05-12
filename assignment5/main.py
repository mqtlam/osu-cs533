import numpy as np

from mdp import MDP
from parking_mdp import ParkingMDP, ParkingAction
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

# run simulation
simulator = MDPSimulator(mdp, initial_state=0)
while not simulator.in_terminal_state():
    # get state and reward
    state = simulator.get_current_state()
    reward = simulator.get_reward()
    print "State {0}".format(print_state_helper(mdp.get_state_params(state)))
    print "Reward {0}".format(reward)
    print

    # random policy
    action = np.random.randint(0, mdp.m)
    print "Action {0}".format(ParkingAction.strings[action])
    print

    # take action
    simulator.take_action(action)
