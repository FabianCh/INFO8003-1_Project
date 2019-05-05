from agent.agent import Agent
from policy.function_policy import FunctionPolicy

import numpy


class ParamtricQIteratoin(Agent):

    def __init__(self, buffer, estimator, maximizer):
        super(ParamtricQIteratoin, self).__init__(buffer, estimator, maximizer)

        self.Q = self.estimator()
        self.parameter = 0
        self.target_network = self.Q
        self.target_network_parameter = self.parameter
        self.learning_ratio = 0.05

    def train(self, depth=100, dataset_size=25):
        assert not self.buffer.is_empty(), print("Empty Buffer")

        for i in range(depth):
            # collect the dataset
            print("Collecting dataset...")
            dataset = self.buffer.get_sample(dataset_size)
            print('Dataset collected.\n')

            # formatting the dataset
            print('formatting the dataset...')
            train_x = []
            train_y = []

            if i == 0:
                for data in dataset:
                    x = [data[0][0], data[0][1], data[0][2], data[0][3], self.parameter, data[1][0], data[1][1]]
                    y = data[2]
                    train_x.append(x)
                    train_y.append(y)
            else:
                for data in dataset:
                    x = [data[0][0], data[0][1], data[0][2], data[0][3], self.parameter, data[1][0], data[1][1]]
                    x_prime = [data[3][0], data[3][1], data[3][2], data[3][3], self.parameter]
                    temporal_difference = data[2] + \
                        self.gamma * self.maximizer.value(self.Q, x_prime) - self.Q(x[:5], x[5:])
                    train_x.append(x)
                    train_y.append(self.Q(x[:5], x[5:]) + self.learning_ratio * temporal_difference)

                    self.parameter = self.parameter + self.learning_ratio * temporal_difference * self.q_derivative(x)

            print('dataset formatted.')

            # Learning the q function
            print("learning Q" + str(i) + "...")
            self.Q.train(train_x, train_y)
            print("function Q" + str(i) + " learnt.\n")

    def q_derivative(self, x):
        x_1, x_2 = x[:4] + [(x[4] + 0.0001)], x[:4] + [(x[4] - 0.0001)]
        y_1, y_2 = self.Q(x_1, x[5:]), self.Q(x_2, x[5:])
        return (y_2 - y_1) / 2 * 0.0001

    def get_optimal_policy(self):
        p = self.parameter

        def foo(state, action):
            new_state = numpy.array([i for i in state] + [p])
            return self.Q(new_state, action)

        return FunctionPolicy(foo, self.maximizer)
