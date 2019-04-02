from agent import Agent
from policy import *


agent = Agent()
action = [4, 0]
static_policy = StaticPolicy(action)

agent.play(static_policy)
print(agent.expected_return(static_policy))
