from agent.fitted_q_iteration import FittedQIteration
from agent.DQNAgent import DQNAgent
from agent.DDQNAgent import DDQNAgent
from buffer.ordered_buffer import OrderedBuffer
from buffer.random_buffer import RandomBuffer
from buffer.priority_buffer import PriorityBuffer
from estimator.linear_regressor_estimator import LinearRegressorEstimator
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator
from estimator.neural_network import NeuralNetworkEstimator
from maximizer.static_sampler import StaticSampler
from maximizer.uniform_sampler import UniformSampler
from maximizer.balanced_uniform_sampler import BalancedUniformSampler
from policy.random_policy import RandomPolicy

import configparser
import csv
import pickle

config = configparser.ConfigParser()
config.read('config3.ini')

# region Maximizer Initialisation
MaximizerType = config['Agent']['Maximizer']
if MaximizerType == 'Static':
    NumberOfSample = int(config['Static']['NumberOfSample'])
    Maximizer = StaticSampler(NumberOfSample, 1.3)
elif MaximizerType == 'Uniform':
    NumberOfSample = int(config['Uniform']['NumberOfSample'])
    Maximizer = UniformSampler(NumberOfSample, 1.3)
elif MaximizerType == 'BalancedUniform':
    NumberOfSample = int(config['BalancedUniform']['NumberOfSample'])
    Maximizer = BalancedUniformSampler(NumberOfSample, 1.3)
else:
    raise ValueError('Maximizer unknown')
# endregion

# region logfile name
prefix2 = 'output/evaluation/'
prefix = ""
for part in ['Train', 'Agent', MaximizerType]:
    prefix += part + '--'
    for param in config[part]:
        prefix += param + '_' + config[part][param] + '-'
# endregion

with open(prefix2 + prefix + ".pickle", 'rb') as file:
    agent = pickle.load(file)

agent.set_policy(agent.get_greedy_policy())

for i in range(25):
    print("\n\nEpisode " + str(i))

    agent.play_and_train(iteration_number=1000)

    agent.set_policy(agent.get_optimal_policy())
    expected_reward, mean_hits = agent.expected_return_and_hit()
    print("\nExpected return : " + str((expected_reward, mean_hits)))
    with open(prefix2 + prefix + "_log.csv", 'a', newline='\n') as log_file:
        csv_writer = csv.writer(log_file, delimiter=';')
        csv_writer.writerow([expected_reward, mean_hits])

    agent.set_policy(agent.get_greedy_policy())

    with open(prefix2 + prefix + '.pickle', 'wb') as agent_file:
        pickle.dump(agent, agent_file, pickle.HIGHEST_PROTOCOL)
        print("Agent saved")

# with open(prefix2 + prefix + "_log.csv", 'a', newline='\n') as log_file:
#     csv_writer = csv.writer(log_file, delimiter=';')
#     agent.set_policy(agent.get_optimal_policy())
#     expected_reward, mean_hits = agent.expected_return_and_hit(1000)
#     csv_writer.writerow([expected_reward, mean_hits])
#
# agent.show(prefix2 + prefix[:15:3] + "_animation")
# endregion
