import random

class Bandit:
    """Bandit problem abstract class.
    """
    def __init__(self, num_arms):
        """Initialization.

        Args:
            num_arms: number of arms for bandit problem
        """
        self.num_arms = num_arms

    def get_num_arms(self):
        """Get the number of arms.

        Returns:
            number of arms
        """
        return self.num_arms

    def pull(self, arm):
        """Pull arm.

        Args:
            arm: arm index

        Returns:
            reward from pulling arm
        """
        raise NotImplementedError

class SBRDBandit(Bandit):
    """Scaled binomial reward distribution (SBRD) bandit problem.
    """
    def __init__(self, arm_params):
        """Initialization.

        Args:
            arm_params: list of parameter tuples for each arm [(r_a, p_a)]
                arm_params[a] gives tuple (r, p) for arm a where
                    r is the [0,1] reward and
                    p is the probability of reward for SBRD
        """
        Bandit.__init__(self, len(arm_params))
        self.arm_params = arm_params

    def pull(self, arm):
        """Pull arm. Implements SBRD.

        Args:
            arm: arm index

        Returns:
            reward from pulling arm
        """
        r, p = self.arm_params[arm]
        success = random.uniform(0, 1) <= p
        return r if success else 0
