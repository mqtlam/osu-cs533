from plot import Plot
from regret import Regret
import time

### PART 1

from bandit import SBRDBandit
from bandit_algorithms import IncrementalUniformAlgorithm
from bandit_algorithms import UCBAlgorithm
from bandit_algorithms import EpsilonGreedyAlgorithm

### PART 2

# bandit problem 1
arm_params = []
for a in range(9):
    arm_params.append((0.05, 1))
arm_params.append((1, 0.1))
bandit1 = SBRDBandit(arm_params, 'bandit1')

# bandit problem 2
arm_params = []
for a in range(20):
    arm_params.append((1.*(a+1)/20, 0.1))
bandit2 = SBRDBandit(arm_params, 'bandit2')

# custom bandit problem
arm_params = [(1, 0.5)]
for a in range(9):
    arm_params.append((0.05, 0.5))
bandit3 = SBRDBandit(arm_params, 'custom_bandit')

### PARTS 3 and 4
num_pulls = 100001
num_trials = 1000

def run_bandit_experiment(bandit, num_pulls, num_trials):
    # specify bandit algorithms below
    algorithm1 = IncrementalUniformAlgorithm(bandit)
    algorithm2 = UCBAlgorithm(bandit)
    algorithm3 = EpsilonGreedyAlgorithm(bandit)
    algorithms = [algorithm1, algorithm2, algorithm3]

    # keep track of data for plotting
    plot_sample_rate = 1
    plot = Plot(num_pulls, num_trials,
        [a.get_name() for a in algorithms], plot_sample_rate)

    # experiment loop
    for a in algorithms:
        print '\nRunning algorithm {}...'.format(a.get_name())
        plot.reset_trial()

        for t in range(num_trials):
            print 'Running trial {}...'.format(t)
            start = time.time()

            plot.begin_trial()
            optimal_expected_reward = bandit.get_expected_reward_optimal_arm()
            regret = Regret(optimal_expected_reward)
            a.reset(bandit)

            for i in range(num_pulls):
                # pull arm according to algorithm
                pulled_arm, _ = a.pull()

                # update regrets
                best_arm = a.get_best_arm()
                expected_reward_pulled_arm = bandit.get_expected_reward_arm(pulled_arm)
                expected_reward_best_arm = bandit.get_expected_reward_arm(best_arm)
                regret.add(expected_reward_pulled_arm, expected_reward_best_arm)

                # update plot
                if i % plot_sample_rate == 0:
                    plot.add_point(i, regret.get_simple_regret(),
                        regret.get_cumulative_regret(), a.get_name())

            end = time.time()
            print '\telapsed: {}'.format(end-start)
            print '\tbest arm: {}'.format(a.get_best_arm())

    # create plot
    plot.plot_simple_regret(bandit.get_name())
    plot.plot_cumulative_regret(bandit.get_name())

    # save
    plot.save('{}_data'.format(bandit.get_name()))

# run bandit experiments
run_bandit_experiment(bandit1, num_pulls, num_trials)
run_bandit_experiment(bandit2, num_pulls, num_trials)
# run_bandit_experiment(bandit3, num_pulls, num_trials)
