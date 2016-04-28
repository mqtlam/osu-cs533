class Regret:
    """Regret keeps track of regret computations.
    """
    def __init__(self, optimal_expected_reward):
        """Initialization.

        Args:
            optimal_expected_reward: optimal expected reward
        """
        self.optimal_expected_reward = optimal_expected_reward
        self.expected_reward_best_arm = 0
        self.expected_cumulative_reward = 0
        self.n = 0

    def add(self, expected_reward_pulled_arm, expected_reward_best_arm):
        """Add regret data.

        Args:
            expected_reward_pulled_arm: expected reward of the latest pulled arm
            expected_reward_best_arm: expected reward of current best arm
        """
        self.expected_reward_best_arm = expected_reward_best_arm
        self.expected_cumulative_reward += expected_reward_pulled_arm
        self.n += 1

    def get_cumulative_regret(self):
        """Get cumulative regret.

        Returns:
            cumulative regret
        """
        return self.n * self.optimal_expected_reward - self.expected_cumulative_reward

    def get_simple_regret(self):
        """Get simple regret.

        Returns:
            simple regret
        """
        return self.optimal_expected_reward - self.expected_reward_best_arm
