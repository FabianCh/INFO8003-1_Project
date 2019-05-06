from agent.fitted_q_iteration import FittedQIteration
from agent.parametric_q_iteration import ParamtricQIteratoin
from buffer.ordered_buffer import OrderedBuffer
from buffer.priority_buffer import PriorityBuffer
from buffer.random_buffer import RandomBuffer
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator
from max_finder.uniform_sampler import UniformSampler
from max_finder.static_sampler import StaticSampler
from policy.random_policy import RandomPolicy
from policy.static_policy import StaticPolicy

# static_policy = StaticPolicy((4, 0))
# random_policy = RandomPolicy()
#
# uniform_sampler = UniformSampler(5, 1.3)
# agent = FittedQIteration(RandomBuffer(), ExtremelyRandomizeTreeEstimator, uniform_sampler)
#
#
# print('Generating buffer...')
# agent.generate_one_step_transition(random_policy, 1000)
# print('Buffer generated\n')
#
#
# agent.train(200, 500)
# optimal_policy = agent.get_optimal_policy()
#
# cumulated_reward, number_of_action = agent.play(optimal_policy)
# print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)
#
# print("Expected return : ", agent.expected_return(optimal_policy))
# agent.show(optimal_policy)
#
#
from agent.DDQNAgent import DDQNAgent

agent = DDQNAgent(PriorityBuffer(), UniformSampler(5))
agent.PlayDQN(1000000)
agent.expected_return(agent.get_optimal_policy(), 1000)
agent.show(agent.get_optimal_policy())
