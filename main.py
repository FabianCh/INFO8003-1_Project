from agent import Agent
from policy import *


agent = Agent()
action = [4, 0]
static_policy = StaticPolicy(action)


cumulated_reward, number_of_action = agent.play(static_policy)
print("Game over, cumulated reward =", cumulated_reward, "number of action =", number_of_action)
print("Expected retrun : ", agent.expected_return(static_policy))
