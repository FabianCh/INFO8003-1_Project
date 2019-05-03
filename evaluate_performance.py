import configparser
from agent.fitted_q_iteration import FittedQIteration
from buffer.ordered_buffer import OrderedBuffer
from buffer.random_buffer import RandomBuffer
from buffer.priority_buffer import PriorityBuffer
from estimator.linear_regressor_estimator import LinearRegressorEstimator
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator
from estimator.neural_network import NeuralNetworkEstimator
from max_finder.uniform_sampler import UniformSampler
from policy.random_policy import RandomPolicy

config = configparser.ConfigParser()
config.read('config.ini')

# region Agent_Initialisation
BufferType = config['Agent']['Buffer']
if BufferType == 'Ordered':
    buffer = OrderedBuffer()
elif BufferType == 'Random':
    buffer = RandomBuffer()
elif BufferType == 'Prioritized':
    buffer = PriorityBuffer()
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
if MaximizerType == 'UniformSampling':
    NumberOfSample = int(config['UniformSampling']['NumberOfSample'])
    Maximizer = UniformSampler(NumberOfSample, 1.3)
else:
    raise ValueError('Maximizer unknown')

agent = FittedQIteration(buffer, Estimator, Maximizer)
# endregion

random_policy = RandomPolicy()
print('Generating buffer...')
NumberOfOST = int(config['Train']['NumberOfOST'])
agent.generate_one_step_transition(random_policy, NumberOfOST)
print('Buffer generated\n')

Depth = int(config['Train']['Depth'])
DatasetSize = int(config['Train']['DatasetSize'])

agent.train(Depth, DatasetSize)
optimal_policy = agent.get_optimal_policy()

print("Expected return of the agent: ", agent.expected_return(optimal_policy))




