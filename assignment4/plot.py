import matplotlib.pyplot as plt
import numpy as np

class Plot:
    """Plot helps keep track of data points and generate plots.
    """
    def __init__(self, num_arm_pulls, num_trials, algorithms, sample_rate=100):
        """Initialization.

        Args:
            num_arm_pulls: number of arm pulls (integer)
            num_trials: number of trials (integer)
            algorithms: list of algorithms (list of strings)
            sample_rate: sample rate for plotting (integer)
        """
        num_points = num_arm_pulls/sample_rate # integer division
        self.arm_pulls = {}
        self.simple_regret = {}
        self.cumulative_regret = {}
        for alg in algorithms:
            self.arm_pulls[alg] = np.zeros((num_points, num_trials))
            self.simple_regret[alg] = np.zeros((num_points, num_trials))
            self.cumulative_regret[alg] = np.zeros((num_points, num_trials))

        self.reset_trial()

    def reset_trial(self):
        """Reset trial.
        """
        self.arm_pulls_counter = 0
        self.trial_counter = -1

    def begin_trial(self):
        """Begin a new trial.
        """
        self.arm_pulls_counter = 0
        self.trial_counter += 1

    def add_point(self, num_arm_pulls, simple_regret, cumulative_regret, algorithm):
        """Add data point.

        Args:
            num_arm_pulls: number of arm pulls (integer)
            simple_regret: simple regret (double)
            cumulative_regret: cumulative_regret (double)
            algorithm: algorithm name (string)
        """
        self.arm_pulls[algorithm][self.arm_pulls_counter, self.trial_counter] = num_arm_pulls
        self.simple_regret[algorithm][self.arm_pulls_counter, self.trial_counter] = simple_regret
        self.cumulative_regret[algorithm][self.arm_pulls_counter, self.trial_counter] = cumulative_regret
        self.arm_pulls_counter = 0

    def plot_simple_regret(self, experiment_name):
        """Plot simple regret and save figure.

        Args:
            experiment_name: experiment name (string)
        """
        self._plot('Simple Regret', self.arm_pulls, self.simple_regret,
            '{}_simple_regret.png'.format(experiment_name))

    def plot_cumulative_regret(self, experiment_name):
        """Plot cumulative regret and save figure.

        Args:
            experiment_name: experiment name (string)
        """
        self._plot('Cumulative Regret', self.arm_pulls, self.cumulative_regret,
            '{}_cumulative_regret.png'.format(experiment_name))

    def _plot(self, regret_type, x, y, output_file):
        """Plot helper. Saves plot to output file.

        Args:
            regret_type: "Simple Regret" or "Cumulative Regret"
            x: dictionary 'algorithm name' -> np.array(num_pulls, num_trials)
            y: dictionary 'algorithm name' -> np.array(num_pulls, num_trials)
            output_file: output file name (string)
        """
        # plot
        color_list = ['blue', 'red', 'green']
        for i, algorithm in enumerate(x):
            plt.plot(np.mean(x[algorithm], axis=1), np.mean(y[algorithm], axis=1),
                color=color_list[i], linewidth=2.5,
                linestyle='-', label=algorithm)
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('Number of Arm Pulls')
        plt.ylabel(regret_type)
        plt.title('{} vs. Number of Arm Pulls'.format(regret_type))

        # save figure
        plt.savefig(output_file, dpi=72)
        plt.clf()
