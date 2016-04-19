from mdp import MDP
from mdp_optimization import InfiniteHorizonPolicyEvaluation
from mdp_optimization import InfiniteHorizonPolicyOptimization

def print_value_function(a):
    print '\n'.join('{0:0.8f}'.format(float(row)) for row in a)

def print_policy(a):
    print '\n'.join('{0}'.format(int(row)) for row in a)

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

# PROBLEM 2
mdp = MDP()

# load MDP1
mdp.load_from_file('MDP1.txt')

epsilon = 0.000001

# run finite horizon value iteration
beta = 0.1
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP1 value iteration, beta={}, epsilon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP1 policy iteration, beta={}".format(beta))

# run finite horizon value iteration
beta = 0.9
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP1 value iteration, beta={}, epsilon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP1 policy iteration, beta={}".format(beta))

# load MDP2
mdp.load_from_file('MDP2.txt')

# run finite horizon value iteration
beta = 0.1
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP2 value iteration, beta={}, epislon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP2 policy iteration, beta={}".format(beta))

# run finite horizon value iteration
beta = 0.9
(V, policy) = InfiniteHorizonPolicyOptimization.value_iteration(mdp, beta, epsilon)
print_helper(V, policy, "MDP2 value iteration, beta={}, epsilon={}".format(beta, epsilon))
(V, policy) = InfiniteHorizonPolicyOptimization.policy_iteration(mdp, beta)
print_helper(V, policy, "MDP2 policy iteration, beta={}".format(beta))
