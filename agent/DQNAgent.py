from agent.agent import Agent
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from policy.function_policy import FunctionPolicy
from estimator.neural_network import NeuralNetworkEstimator


class DQNAgent(Agent):

    def __init__(self, buffer, maximizer,
                 decrease_rate=0.000001,
                 target_network_update=10000):
        super(DQNAgent, self).__init__(NeuralNetworkEstimator, buffer, maximizer)

        self.Q = self.estimator()
        self.target_Q = self.estimator()
        self.target_Q.copy(self.Q)

        self.epsilon = 1
        self.decrease_rate = decrease_rate

        self.target_network_update = target_network_update
        self.actions_taken = 0

    def reinitialize_actions(self):
        self.actions_taken = 0

    def play_and_train(self, iteration_number=1000):
        state, is_terminate = self.domain.observe(), False

        action_number = 1
        while action_number < iteration_number:
            if is_terminate:
                self.reinitialize_domain()
                state = self.domain.observe()
                print('End')

            action = self.policy(state)
            next_state, reward, is_terminate = self.domain.step(action)

            self.actions_taken += 1
            action_number += 1

            if reward == 3:
                print('Hit')
            elif reward == -1:
                print('Miss')

            one_step_transition = (state, action, reward, next_state)
            self.buffer.add_sample(one_step_transition)

            self.train()

            if self.actions_taken % self.target_network_update == 0:
                self.target_Q.copy(self.Q)
            if (action_number * 10 % iteration_number) == 0:
                print(action_number * 100 // iteration_number, '%')
            state = next_state

    def train(self, epoch=1, mini_batch=25):
        dataset = self.buffer.get_sample(mini_batch)

        train_x = []
        train_y = []

        for state, action, reward, next_state in dataset:
            x = [state[0], state[1], state[2], state[3], action[0], action[1]]
            x_prime = [next_state[0], next_state[1], next_state[2], next_state[3]]
            y = reward + self.gamma * self.maximizer.value(self.target_Q, x_prime)
            train_x.append(x)
            train_y.append(y)
        self.Q.train(train_x, train_y)

    def get_optimal_policy(self):
        return FunctionPolicy(self.Q, self.maximizer)

    def get_greedy_policy(self):
        result = GreedyPolicy(self.Q, self.maximizer, self.epsilon)

        self.epsilon -= self.decrease_rate
        if self.epsilon < 0.01:
            self.epsilon = 0.01

        return result
