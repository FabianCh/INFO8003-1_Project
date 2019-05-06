from agent.DDQNAgent import DDQNAgent
from buffer.priority_buffer import PriorityBuffer
from maximizer.uniform_sampler import UniformSampler

agent = DDQNAgent(PriorityBuffer(), UniformSampler(11))
agent.play_and_train(300000)
agent.expected_return(agent.get_optimal_policy(), 1000)
agent.show(agent.get_optimal_policy())
