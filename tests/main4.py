from agent.DDQNAgent import DDQNAgent
from buffer.random_buffer import RandomBuffer
from maximizer.static_sampler import StaticSampler

agent = DDQNAgent(RandomBuffer(), StaticSampler(11))
agent.play_and_train(300000)
agent.expected_return(agent.get_optimal_policy(), 1000)
agent.show(agent.get_optimal_policy())
