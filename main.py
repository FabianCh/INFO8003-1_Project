from agent import Agent
from policy import *


agent = Agent()
action = [4, 0]
static_policy = StaticPolicy(action)


cumulated_reward, number_of_action = agent.play(static_policy)
print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)
print("Expected return : ", agent.expected_return(static_policy), "\n")

print('Generating buffer...')
random_policy = RandomPolicy()
agent.generate_one_step_transition(random_policy, 2000)
print('buffer generated\n')
agent.fitted_q_iteration(30)
optimal_policy = OptimalPolicy(agent.approximation_function_Qn[-1])

cumulated_reward, number_of_action = agent.play(optimal_policy)
print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)
print("Expected return : ", agent.expected_return(optimal_policy))
