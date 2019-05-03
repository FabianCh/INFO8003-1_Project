from agent.fitted_q_iteration import FittedQIteration
from agent.parametric_q_iteration import ParamtricQIteratoin
from buffer.ordered_buffer import OrderedBuffer
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator
from max_finder.uniform_sampler import UniformSampler
from policy.random_policy import RandomPolicy
from policy.static_policy import StaticPolicy

static_policy = StaticPolicy((4, 0))
random_policy = RandomPolicy()

uniform_sampler = UniformSampler(5, 1.3)
agent = ParamtricQIteratoin(OrderedBuffer(), ExtremelyRandomizeTreeEstimator, uniform_sampler)


print('Generating buffer...')
agent.generate_one_step_transition(random_policy, 100)
print('Buffer generated\n')


agent.train(25, 100)
optimal_policy = agent.get_optimal_policy()

cumulated_reward, number_of_action = agent.play(optimal_policy)
print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)

print("Expected return : ", agent.expected_return(optimal_policy))
agent.show(optimal_policy)
