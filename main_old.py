from agent.fitted_q_iteration import FittedQIteration
from buffer.ordered_buffer import OrderedBuffer
from buffer.priority_buffer import PriorityBuffer
from buffer.random_buffer import RandomBuffer
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator
from maximizer.uniform_sampler import UniformSampler
from maximizer.static_sampler import StaticSampler
from maximizer.balanced_uniform_sampler import BalancedUniformSampler
from policy.random_policy import RandomPolicy
from policy.static_policy import StaticPolicy

import pickle
with open('output/evaluation/Train--initialbuffersize_50000-episode_300-episode_size_1000-minibatchsize_30-decreaserate_0.00001-targetupdate_10000-Agent--agent_DDQNAgent-estimator_NeuralNetwork-buffer_Prioritized-maximizer_Uniform-Uniform--numberofsample_11-.pickle', 'rb') as file:
    agent = pickle.load(file)
    agent.set_policy(agent.get_optimal_policy())
    agent.show(verbose=True)
    agent.expected_return_and_hit(verbose=True)
