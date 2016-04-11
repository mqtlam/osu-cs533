from mdp import MDP
from mdp_optimization import MDPOptimization

def print_value_function(a):
    print '\n'.join('\t'.join('{0:0.4f}'.format(float(cell)) for cell in row) for row in a)

def print_policy(a):
    print '\n'.join('\t'.join('{0}'.format(int(cell)) for cell in row) for row in a)

def print_helper(V, policy, name):
    print "============================================================"
    print
    print "=== {} ===".format(name)
    print "non-stationary value function:"
    print_value_function(V)
    print
    print "policy:"
    print_policy(policy)
    print
    print "============================================================"
    print
    print

# PROBLEM 1

# load MDP debug
mdp = MDP()
mdp.load_from_file('MDP_debug.txt')

# run finite horizon value iteration
H = 10
(V, policy) = MDPOptimization.finite_horizon_value_iteration(mdp, H)
print_helper(V, policy, "MDP Debug")

# PROBLEM 2

# load custom MDP
mdp = MDP()
mdp.load_from_file('MDP_custom.txt')

# run finite horizon value iteration
H = 10
(V, policy) = MDPOptimization.finite_horizon_value_iteration(mdp, H)
print_helper(V, policy, "MDP Custom")

# PROBLEM 3

# load MDP1
mdp.load_from_file('MDP1.txt')

# run finite horizon value iteration
H = 10
(V, policy) = MDPOptimization.finite_horizon_value_iteration(mdp, H)
print_helper(V, policy, "MDP1")

# load MDP1
mdp.load_from_file('MDP2.txt')

# run finite horizon value iteration
H = 10
(V, policy) = MDPOptimization.finite_horizon_value_iteration(mdp, H)
print_helper(V, policy, "MDP2")
