from mdp import MDP
from mdp_optimization import MDPOptimization

def print_helper(V, policy, name):
    print "=== {} ===".format(name)
    print "non-stationary value function:"
    print V
    print
    print "policy:"
    print policy
    print

# load MDP debug
mdp = MDP()
mdp.load_from_file('MDP_debug.txt')

# run finite horizon value iteration
H = 10
(V, policy) = MDPOptimization.finite_horizon_value_iteration(mdp, H)
print_helper(V, policy, "MDP Debug")

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
