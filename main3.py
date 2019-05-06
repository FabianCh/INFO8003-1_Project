from agent.DDQNAgent import DDQNAgent
from buffer.priority_buffer import PriorityBuffer
from max_finder.balanced_uniform_sampler import BalancedUniformSampler

agent = DDQNAgent(PriorityBuffer(), BalancedUniformSampler(5))
agent.PlayDQN(300000)
agent.expected_return(agent.get_optimal_policy(), 1000)
agent.show(agent.get_optimal_policy())
