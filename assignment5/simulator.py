import numpy as np

from mdp import MDP

class MDPSimulator:
    def __init__(self, mdp, initial_state=0):
        self.mdp = mdp
        self.current_state = initial_state
        self.terminal_state = False

    def take_action(self, action):
        if self.terminal_state:
            return

        # set up distribution to sample
        values = np.array(range(self.mdp.n))
        probabilities = self.mdp.get_transition_prob(self.current_state, action)
        if np.sum(probabilities) == 0:
            return
        bins = np.add.accumulate(probabilities)

        # sample a state from the distribution and update
        next_state = values[np.digitize(np.random.random_sample(1), bins)[0]]
        self.current_state = next_state

        # check if in terminal state
        self.terminal_state = True
        for a in range(self.mdp.m):
            probabilities = self.mdp.get_transition_prob(self.current_state, a)
            if np.sum(probabilities) > 0:
                self.terminal_state = False
                break

    def in_terminal_state(self):
        return self.terminal_state

    def get_current_state(self):
        return self.current_state

    def get_reward(self):
        return self.mdp.get_reward(self.current_state)
