from agent import Agent
from buffer.random_buffer import RandomBuffer
from buffer.priority_buffer import PriorityBuffer
from buffer.ordered_buffer import OrderedBuffer
from estimator.linear_regressor_estimator import  LinearRegressorEstimator
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator
from policy import *
import catcher

agent = Agent(OrderedBuffer(), ExtremelyRandomizeTreeEstimator)

action = [4, 0]
static_policy = StaticPolicy(action)
random_policy = RandomPolicy()

# domain = catcher.ContinuousCatcher()
# print(domain.bar.size)
# for i in range(10):
#     cumulated_reward, number_of_action = agent.play(random_policy)
#     print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)
# cumulated_reward, number_of_action = agent.show(random_policy)
# print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)
# print("Expected return : ", agent.expected_return(static_policy), "\n")


print('Generating buffer...')
agent.generate_one_step_transition(random_policy, 1000)
print('buffer generated\n')

agent.fitted_q_iteration(50)
optimal_policy = OptimalPolicy(agent.approximation_function_Qn[-1])

cumulated_reward, number_of_action = agent.play(optimal_policy)
print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)

print("Expected return : ", agent.expected_return(optimal_policy))
agent.show(optimal_policy)

# print(agent.approximation_function_Qn[-1]._coeff)