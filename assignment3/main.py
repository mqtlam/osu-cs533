from mdp import MDP
from parking_mdp import ParkingMDP, ParkingAction
from mdp_optimization import InfiniteHorizonPolicyEvaluation
from mdp_optimization import InfiniteHorizonPolicyOptimization

### helper functions to print stuff

def print_value_function(a):
    print '\n'.join('{0:0.8f}'.format(float(row)) for row in a)

def print_policy(a):
    print '\n'.join('{0}'.format(int(row)) for row in a)

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

def print_parking_value_function(V, mdp):
    for s in range(mdp.n):
        print "{0}\t\t{1:0.8f}".format(print_state_helper(mdp.get_state_params(s)), float(V[s]))

def print_parking_policy(policy, mdp):
    for s in range(mdp.n):
        p = ParkingAction.strings[policy[s]]
        print "{0}\t\t{1}".format(print_state_helper(mdp.get_state_params(s)), p)

def print_helper(V, policy, name):
    print "============================================================"
    print
    print "=== {} ===".format(name)
    print "value function:"
    print_value_function(V)
    print
    print "policy:"
    print_policy(policy)
    print
    print "============================================================"
    print
    print

def print_parking_helper(V, policy, mdp, name):
    print "============================================================"
    print
    print "=== {} ===".format(name)
    print "value function:"
    print_parking_value_function(V, mdp)
    print
    print "policy:"
    print_parking_policy(policy, mdp)
    print
    print "============================================================"
    print
    print

### PROBLEM 2
mdp = MDP()

# load MDP1
mdp.load_from_file('MDP1.txt')

epsilon = 0.000001

# run infinite horizon value iteration and policy iteration
beta = 0.1
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP1 value iteration, beta={}, epsilon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP1 policy iteration, beta={}".format(beta))

# run infinite horizon value iteration and policy iteration
beta = 0.9
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP1 value iteration, beta={}, epsilon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP1 policy iteration, beta={}".format(beta))

# load MDP2
mdp.load_from_file('MDP2.txt')

# run infinite horizon value iteration and policy iteration
beta = 0.1
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP2 value iteration, beta={}, epislon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP2 policy iteration, beta={}".format(beta))

# run infinite horizon value iteration and policy iteration
beta = 0.9
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP2 value iteration, beta={}, epsilon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP2 policy iteration, beta={}".format(beta))

### PROBLEM 3
# first version
mdp = ParkingMDP(10, handicap_reward=-100, collision_reward=-10000)
mdp.save_to_file('MDP_parking1.txt')

# run policy iteration
beta = 0.99
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_parking_helper(V, policy, mdp, "Parking MDP policy iteration, beta={}".format(beta))

# second version
mdp = ParkingMDP(10, handicap_reward=100)
mdp.save_to_file('MDP_parking2.txt')

# run policy iteration
beta = 0.99
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_parking_helper(V, policy, mdp, "Parking MDP policy iteration, beta={}".format(beta))
