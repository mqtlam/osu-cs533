# This is a hacky way of re-plotting graphs...

from plot import Plot
from bandit_algorithms import IncrementalUniformAlgorithm
from bandit_algorithms import UCBAlgorithm
from bandit_algorithms import EpsilonGreedyAlgorithm
from bandit import SBRDBandit

# load old plot
arm_params = [(1,1)] # dummy params
b = SBRDBandit(arm_params, 'custom_bandit')

num_pulls = 10001
num_trials = 1000
plot_sample_rate = 1
algorithms = [IncrementalUniformAlgorithm(b), UCBAlgorithm(b), EpsilonGreedyAlgorithm(b)]
plot = Plot(num_pulls, num_trials, [a.get_name() for a in algorithms], plot_sample_rate)

print "loading data..."
plot.load('custom_bandit_data.npz')

# new plot
print "creating plots..."
sample_rate = 1
end_index = 501
plot.plot_cumulative_regret('new_'+b.get_name(), sample_rate, end_index)
plot.plot_simple_regret('new_'+b.get_name(), sample_rate, end_index)
