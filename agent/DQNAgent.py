from agent.agent import Agent
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from estimator.neural_network import NeuralNetworkEstimator


# def copyModel2Model(model_source,model_target,certain_layer=""):
#     for l_tg,l_sr in zip(model_target.layers,model_source.layers):
#         wk0=l_sr.get_weights()
#         l_tg.set_weights(wk0)
#         if l_tg.name==certain_layer:
#             break
#     print("model source was copied into model target")


class DQNAgent(Agent):

    def __init__(self, buffer, maximazer):
        super(DQNAgent, self).__init__(buffer, NeuralNetworkEstimator, maximazer)

        self.Q = self.estimator()
        self.target = self.estimator()

    def PlayDQN(self, T):
        self.domain.reset()
        policy = RandomPolicy()

        state, is_terminate = self.domain.observe(), False



        t = 1
        while t < T:
            if is_terminate:
                self.domain.reset()
                state = self.domain.observe()
                print('end')

            action = policy(state)
            next_state, reward, is_terminate = self.domain.step(action)
            if reward == 3:
                print('hit')

            one_step_transition = (state, action, reward, next_state)
            self.buffer.add_sample(one_step_transition)

            self.train(1)

            state = next_state
            t += 1
            # print(t)

    def train(self, depth=1, mini_batch=30):
        list_ost = self.buffer.get_sample(mini_batch)

        train_x = []
        train_y = []

        for state, action, reward, next_state in list_ost:
            if reward == 3:
                print("    Learn From Hit")
            x = [state[0], state[1], state[2], state[3], action[0], action[1]]
            x_prime = [next_state[0], next_state[1], next_state[2], next_state[3]]
            y = reward + \
                self.gamma * self.maximizer.value(self.Q, x_prime)
            train_x.append(x)
            train_y.append(y)

        self.Q.train(train_x, train_y)

    def get_optimal_policy(self):
        return GreedyPolicy(self.Q, self.maximizer, 0.1)
