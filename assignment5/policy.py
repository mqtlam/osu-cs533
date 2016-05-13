import numpy as np

from parking_mdp import ParkingAction

class Policy:
    """Policy abstract class
    """
    def get_action(self, state):
        """Get the policy action for the given state.

        Args:
            state: state

        Returns:
            action
        """
        raise NotImplementedError()

class ParkingPolicy(Policy):
    """Parking policy abstract class
    """
    pass

class RandomParkingPolicy(ParkingPolicy):
    """Random policy.

    Selects PARK with probability p and DRIVE with probability 1-p.
    """
    def __init__(self, mdp, park_probability=0.5):
        """Initialization.

        Args:
            mdp: MDP object
            park_probability: probability of parking [0,1]
        """
        self.mdp = mdp
        self.park_probability = park_probability

    def get_action(self, state):
        (column, row, occupied, parked) = self.mdp.get_state_params(state)

        if parked == 1:
            action = ParkingAction.EXIT
        elif np.random.uniform() <= self.park_probability:
            action = ParkingAction.PARK
        else:
            action = ParkingAction.DRIVE

        return action

class SafeRandomParkingPolicy(ParkingPolicy):
    """Safer random policy.

    If occupied, selects DRIVE. Otherwise:
    Selects PARK with probability p and DRIVE with probability 1-p.
    """
    def __init__(self, mdp, park_probability=0.5):
        """Initialization.

        Args:
            mdp: MDP object
            park_probability: probability of parking [0,1]
        """
        self.mdp = mdp
        self.park_probability = park_probability

    def get_action(self, state):
        (column, row, occupied, parked) = self.mdp.get_state_params(state)

        if parked == 1:
            action = ParkingAction.EXIT
        elif occupied == 1:
            action = ParkingAction.DRIVE
        elif occupied == 0 and np.random.uniform() <= self.park_probability:
            action = ParkingAction.PARK
        else:
            action = ParkingAction.DRIVE

        return action
