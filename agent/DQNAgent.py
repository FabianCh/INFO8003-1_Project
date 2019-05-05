from agent.agent import Agent
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from estimator.neural_network import NeuralNetworkEstimator


class DQNAgent(Agent):

    def __init__(self, buffer, maximazer):
        super(DQNAgent, self).__init__(buffer, NeuralNetworkEstimator, maximazer)

        self.Q = self.estimator()
        self.target_Q = self.estimator()
        self.target_Q.copy(self.Q)
        self.learning_ratio = 0.05

    def PlayDQN(self, T):
        self.domain.reset()
        policy = RandomPolicy()
        self.generate_one_step_transition(policy)

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

            self.train()

            if t % 10000 == 0:
                self.target_Q.copy(self.Q)

            state = next_state
            t += 1
            if t % 10000 == 0:
                print(t/1000, '%')

    def train(self, depth=1, mini_batch=30):
        list_ost = self.buffer.get_sample(mini_batch)

        train_x = []
        train_y = []

        for state, action, reward, next_state in list_ost:
            x = [state[0], state[1], state[2], state[3], action[0], action[1]]
            x_prime = [next_state[0], next_state[1], next_state[2], next_state[3]]
            y = reward + \
                self.gamma * self.maximizer.value(self.target_Q, x_prime)
            train_x.append(x)
            train_y.append(y)

        self.Q.train(train_x, train_y)

    def get_optimal_policy(self):
        return GreedyPolicy(self.Q, self.maximizer, 0.1)
