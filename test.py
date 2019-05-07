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
import io
import pickle


config = configparser.ConfigParser()
config.read('config3.ini')


# region Buffer Initialisation
BufferType = config['Agent']['Buffer']
if BufferType == 'Ordered':
    Buffer = OrderedBuffer()
elif BufferType == 'Random':
    Buffer = RandomBuffer()
elif BufferType == 'Prioritized':
    Buffer = PriorityBuffer()
else:
    raise ValueError('Buffer strategy unknown')
# endregion

# region Estimator Initialisation
EstimatorType = config['Agent']['Estimator']
if EstimatorType == 'Linear':
    Estimator = LinearRegressorEstimator
elif EstimatorType == 'RandomTree':
    Estimator = ExtremelyRandomizeTreeEstimator
elif EstimatorType == 'NeuralNetwork':
    Estimator = NeuralNetworkEstimator
else:
    raise ValueError('Estimator unknown')
# endregion

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

# region Initial Buffer Config
InitialBufferSize = config['Train']['InitialBufferSize']
if InitialBufferSize == 'None':
    InitialBufferSize = None
else:
    InitialBufferSize = int(InitialBufferSize)
#endregion

# region Episodes Config
Episode = int(config['Train']['Episode'])
Episode_Size = int(config['Train']['Episode_Size'])
MiniBatchSize = int(config['Train']['MiniBatchSize'])
TargetUpdate = int(config['Train']['TargetUpdate'])
# endregion

# region Agent Initialisation
Agent = config['Agent']['Agent']
if Agent == 'FittedQIteration':
    agent = FittedQIteration(Estimator, Buffer, Maximizer)
elif Agent == 'DQNAgent':
    agent = DQNAgent(Buffer, Maximizer, TargetUpdate)
elif Agent == 'DDQNAgent':
    agent = DDQNAgent(Buffer, Maximizer, TargetUpdate)
else:
    raise ValueError('Agent unknown')
# endregion

# region logfile name
prefix2 = 'output/evaluation/'
prefix = ""
for part in ['Train', 'Agent', MaximizerType]:
    prefix += part + '--'
    for param in config[part]:
        prefix += param + '_' + config[part][param] + '-'
# endregion

# region Evaluation Core
random_policy = RandomPolicy()
agent.set_policy(random_policy)
print('Generating Initilal Buffer...')
agent.generate_one_step_transition(InitialBufferSize)
print('Buffer generated\n')

agent.set_policy(agent.get_greedy_policy())

with open(prefix2 + prefix + "_log.csv", 'w', newline='\n') as log_file:
    csv_writer = csv.writer(log_file, delimiter=';')

for i in range(Episode):
    print("\nEpisode " + str(i))

    agent.play_and_train(iteration_number=Episode_Size)

    agent.set_policy(agent.get_optimal_policy())
    expected_reward, mean_hits = agent.expected_return_and_hit()
    print("Expected return : " + str((expected_reward, mean_hits)))
    with open(prefix2 + prefix + "_log.csv", 'a', newline='\n') as log_file:
        csv_writer = csv.writer(log_file, delimiter=';')
        csv_writer.writerow(['\n', expected_reward, mean_hits])

    agent.set_policy(agent.get_greedy_policy())
    print("\nExpected return Estimated")

    with open(prefix2 + prefix + '.pickle', 'wb') as agent_file:
        pickle.dump(agent, agent_file, pickle.HIGHEST_PROTOCOL)
        print("Agent saved")

    agent.set_policy(agent.get_optimal_policy())
    expected_reward, mean_hits = agent.expected_return_and_hit(1000)
    csv_writer.writerow([expected_reward, mean_hits])

agent.show(prefix2 + prefix[:15:3] + "_animation")
# endregion
