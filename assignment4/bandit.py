import numpy as np
import random

class Bandit:
    """Bandit problem abstract class.
    """
    def __init__(self, num_arms, name='bandit'):
        """Initialization.

        Args:
            num_arms: number of arms for bandit problem
            name: custom name for bandit
        """
        self.num_arms = num_arms
        self.name = name

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

    def get_expected_reward_optimal_arm(self):
        """Get expected reward of the optimal arm.
        For calculating regret.

        Returns:
            expected reward of the optimal arm
        """
        raise NotImplementedError

    def get_expected_reward_arm(self, arm):
        """Get expected reward of arm.
        For calculating regret.

        Args:
            arm: arm index

        Returns:
            expected reward of arm
        """
        raise NotImplementedError

    def get_name(self):
        """Get name of bandit.

        Returns:
            custom name of bandit
        """
        return self.name

class SBRDBandit(Bandit):
    """Scaled binomial reward distribution (SBRD) bandit problem.
    """
    def __init__(self, arm_params, name='bandit'):
        """Initialization.

        Args:
            arm_params: list of parameter tuples for each arm [(r_a, p_a)]
                arm_params[a] gives tuple (r, p) for arm a where
                    r is the [0,1] reward and
                    p is the probability of reward for SBRD
        """
        # input sanity check
        for i, (r, p) in enumerate(arm_params):
            if r < 0. or r > 1.:
                raise ValueError('Invalid r param ({0}) for arm {1}'.format(r, i))
            if p < 0. or p > 1.:
                raise ValueError('Invalid p param ({0}) for arm {1}'.format(p, i))

        # initialize
        Bandit.__init__(self, len(arm_params), name)
        self.arm_params = arm_params
        self.optimal_arm = np.argmax([p*r for (r, p) in arm_params])

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

    def get_expected_reward_optimal_arm(self):
        """Get expected reward of the optimal arm.
        For calculating regret.

        Returns:
            expected reward of the optimal arm
        """
        return self.get_expected_reward_arm(self.optimal_arm)

    def get_expected_reward_arm(self, arm):
        """Get expected reward of arm.
        For calculating regret.

        Args:
            arm: arm index

        Returns:
            expected reward of arm
        """
        (r, p) = self.arm_params[arm]
        return r*p
