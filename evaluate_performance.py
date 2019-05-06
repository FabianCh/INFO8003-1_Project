import configparser
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
import csv

# region Parameter Initialisation
config = configparser.ConfigParser()
config.read('config1.ini')


BufferType = config['Agent']['Buffer']
if BufferType == 'Ordered':
    Buffer = OrderedBuffer()
elif BufferType == 'Random':
    Buffer = RandomBuffer()
elif BufferType == 'Prioritized':
    Buffer = PriorityBuffer()
else:
    raise ValueError('Buffer strategy unknown')

EstimatorType = config['Agent']['Estimator']
if EstimatorType == 'Linear':
    Estimator = LinearRegressorEstimator
elif EstimatorType == 'RandomTree':
    Estimator = ExtremelyRandomizeTreeEstimator
elif EstimatorType == 'NeuralNetwork':
    Estimator = NeuralNetworkEstimator
else:
    raise ValueError('Estimator unknown')

MaximizerType = config['Agent']['Maximizer']

if MaximizerType == 'StaticSampler':
    NumberOfSample = int(config['StaticSampler']['NumberOfSample'])
    Maximizer = StaticSampler(NumberOfSample, 1.3)
elif MaximizerType == 'UniformSampling':
    NumberOfSample = int(config['UniformSampling']['NumberOfSample'])
    Maximizer = UniformSampler(NumberOfSample, 1.3)
elif MaximizerType == 'UniformSampling':
    NumberOfSample = int(config['BalancedUniformSampler']['NumberOfSample'])
    Maximizer = BalancedUniformSampler(NumberOfSample, 1.3)
else:
    raise ValueError('Maximizer unknown')

InitialBufferSize = config['Train']['InitialBufferSize']
if InitialBufferSize == 'None':
    InitialBufferSize = None
else:
    InitialBufferSize = int(InitialBufferSize)

Episode = int(config['Train']['Episode'])
MiniBatchSize = int(config['Train']['MiniBatchSize'])
TargetUpdate = int(config['Train']['TargetUpdate'])
# endregion

# region Agent Initialisation
Agent = config['Agent']['Agent']
if Agent == 'FittedQIteration':
    agent = FittedQIteration(Buffer, Estimator, Maximizer)
elif Agent == 'DQNAgent':
    agent = DQNAgent(Buffer, Maximizer, TargetUpdate)
elif Agent == 'DDQNAgent':
    agent = DDQNAgent(Buffer, Maximizer, TargetUpdate)
else:
    raise ValueError('Agent unknown')
# endregion

# region logfile name
filename = 'output/log/log_'
for part in ['Train', 'Agent', MaximizerType]:
    filename += part + '--'
    for param in config[part]:
        filename += param + '_' + config[part][param] + '-'
filename + '.csv'
# endregion

# region Evaluation Core
random_policy = RandomPolicy()
print('Generating buffer...')
agent.generate_one_step_transition(random_policy, InitialBufferSize)
print('Buffer generated\n')


agent.play_and_train(iteration_number=1000, reset=True)
for _ in range(Episode):
    print("New episode")
    agent.play_and_train(iteration_number=1000, reset=False)
    policy = agent.get_greedy_policy()
    expected_reward, mean_hits = agent.expected_return_and_hit(policy)
    with open(filename, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(expected_reward)

# endregion
