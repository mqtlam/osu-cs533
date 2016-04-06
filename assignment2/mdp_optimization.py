import numpy as np
from mdp import MDP

class MDPOptimization:
    """Policy optimization for MDPs.
    """

    @staticmethod
    def finite_horizon_value_iteration(mdp, H):
        """Finite-horizon value iteration algorithm.

        Args:
            mdp: Markov Decision Process object
            H: time horizon, positive integer

        Returns:
            (optimal non-stationary value function, non-stationary policy)
            for the MDP and time horizon
        """
        V = np.zeros((mdp.n, H))
        policy = np.zeros((mdp.n, H))

        # base case
        for s in range(mdp.n):
            V[s, H-1] = mdp.get_reward(s)

        # recursive case
        for h in reversed(range(H-1)):
            for s in range(mdp.n):
                expectations = [np.dot(mdp.get_transition_prob(s, a), V[:, h+1]) for a in range(mdp.m)]
                V[s, h] = mdp.get_reward(s) + np.max(expectations)
                policy[s, h] = np.argmax(expectations)

        return (V, policy)
