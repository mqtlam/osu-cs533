import numpy as np
import random

class BanditAlgorithm:
    """Bandit algorithm abstract class.
    """
    def __init__(self, bandit):
        """Initialize.

        Args:
            bandit: bandit object
        """
        self.reset(bandit)

    def reset(self, bandit=None):
        """Reset algorithm.

        Args:
            bandit: bandit object
        """
        if bandit is not None:
            self.bandit = bandit
        num_arms = bandit.get_num_arms()
        self.running_sum_rewards = np.zeros(num_arms, dtype=float)
        self.num_pulls = np.zeros(num_arms, dtype=int)
        self.total_pulls = 0
        self.average_rewards = np.zeros(num_arms, dtype=float)
        self.best_arm = random.randint(0, num_arms-1)

    def pull(self):
        """Pull an arm as determined by the algorithm.

        Returns:
            (arm index, reward)
        """
        raise NotImplementedError

    def update(self, reward, arm):
        """Update helper.

        Args:
            reward: reward
            arm: arm index
        """
        # update running sum and pull counts
        self.running_sum_rewards[arm] += reward
        self.num_pulls[arm] += 1
        self.total_pulls += 1

        # update averages
        #old_err_state = np.seterr(divide='ignore', invalid='ignore')
        self.average_rewards = np.divide(self.running_sum_rewards, self.num_pulls)
        self.average_rewards[np.isinf(self.average_rewards)] = 0
        self.average_rewards[np.isnan(self.average_rewards)] = 0

        # update best arm
        self.best_arm = np.argmax(self.get_average_rewards())

    def get_bandit(self):
        """Get bandit.

        Returns:
            bandit object
        """
        return self.bandit

    def get_total_pulls(self):
        """Get the total number of pulls so far.

        Returns:
            total pulls
        """
        return self.total_pulls

    def get_average_rewards(self):
        """Compute the average rewards so far.

        Returns:
            average rewards for each arm
        """
        return self.average_rewards

    def get_best_arm(self):
        """Get the best arm so far.

        Returns:
            best arm index
        """
        return self.best_arm

    def get_name(self):
        """Get name of algorithm.

        Returns:
            name string
        """
        raise NotImplementedError

class IncrementalUniformAlgorithm(BanditAlgorithm):
    """Incremental uniform algorithm.
    """
    def __init__(self, bandit):
        self.reset(bandit)

    def pull(self):
        # pull arm round robin style
        arm = self.get_total_pulls() % self.bandit.get_num_arms()
        reward = self.bandit.pull(arm)

        # update
        self.update(reward, arm)

        return (arm, reward)

    def get_name(self):
        """Get name of algorithm.

        Returns:
            name string
        """
        return "Incremental Uniform"

class UCBAlgorithm(BanditAlgorithm):
    """UCB algorithm.
    """
    def __init__(self, bandit):
        self.reset(bandit)

    def pull(self):
        # decide which arm to pull
        if self.get_total_pulls() < self.bandit.get_num_arms():
            # pull all arms first to avoid divide by zero
            arm = self.get_total_pulls() % self.bandit.get_num_arms()
        else:
            averages = self.get_average_rewards()
            numerator = 2.*np.log(self.get_total_pulls())
            ratio = np.divide(numerator, self.num_pulls)
            exploration_term = np.sqrt(ratio)
            arm = np.argmax(averages + exploration_term)

        # pull arm
        reward = self.bandit.pull(arm)

        # update
        self.update(reward, arm)

        return (arm, reward)

    def get_name(self):
        """Get name of algorithm.

        Returns:
            name string
        """
        return "UCB"

class EpsilonGreedyAlgorithm(BanditAlgorithm):
    """Epsilon greedy algorithm.
    """
    def __init__(self, bandit, epsilon=0.5):
        self.reset(bandit)
        self.epsilon = epsilon

    def pull(self):
        # decide which arm to pull
        num_arms = self.bandit.get_num_arms()
        best_arm = self.get_best_arm()
        if random.uniform(0, 1) <= self.epsilon:
            arm = best_arm
        else:
            other_arms = list(set(range(num_arms)) - set([best_arm]))
            rand_index = random.randint(0, num_arms-2)
            arm = other_arms[rand_index]

        # pull arm
        reward = self.bandit.pull(arm)

        # update
        self.update(reward, arm)

        return (arm, reward)

    def get_name(self):
        """Get name of algorithm.

        Returns:
            name string
        """
        return "Epsilon-Greedy, epsilon={}".format(self.epsilon)
