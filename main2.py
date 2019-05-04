from agent.agent import Agent
from policy.random_policy import RandomPolicy
from buffer.buffer import Buffer
from estimator.estimator import Estimator
from max_finder.uniform_sampler import UniformSampler

agent = Agent(Buffer(), Estimator(), UniformSampler(1,1))

print(agent.expected_return(RandomPolicy(), 1000))
