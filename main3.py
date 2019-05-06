from agent.DDQNAgent import DDQNAgent
from buffer.priority_buffer import PriorityBuffer
from maximizer.balanced_uniform_sampler import BalancedUniformSampler

agent = DDQNAgent(PriorityBuffer(), BalancedUniformSampler(5))
agent.play_and_train(300000)
agent.expected_return(agent.get_optimal_policy(), 1000)
agent.show(agent.get_optimal_policy())
