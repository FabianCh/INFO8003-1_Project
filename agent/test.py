from agent.agent import Agent
from policy.function_policy import FunctionPolicy

import numpy


class TestAgent(Agent):

    def __init__(self, buffer, estimator, maximizer):
        super(TestAgent, self).__init__(buffer, estimator, maximizer)

        self.Q = self.estimator()
        self.parameter = 0
        self.learning_ratio = 0.05

    def train(self, dataset_size=100, epoch=1):
        if self.buffer.is_empty():
            print("Empty Buffer")
            return

        saved_estimator = self.Q
        saved_parameter = self.parameter

        for i in range(epoch):
            # collect the dataset
            print("Collecting dataset...")
            dataset = self.buffer.get_sample(dataset_size)
            print('Dataset collected.\n')

            # formatting the dataset
            print('formatting the dataset...')
            train_x_A = []
            train_x_B = []
            train_y_A = []
            train_y_B = []

            for data in dataset:
                x_A = [data[0][0], data[0][1], data[0][2], data[0][3], self.parameter_A, data[1][0], data[1][1]]
                x_B = [data[0][0], data[0][1], data[0][2], data[0][3], self.parameter_B, data[1][0], data[1][1]]

                x_prime_A = [data[3][0], data[3][1], data[3][2], data[3][3], saved_parameter_A]
                x_prime_B = [data[3][0], data[3][1], data[3][2], data[3][3], saved_parameter_B]

                temporal_difference_A = self.Q_A(x_A[:5], x_A[5:]) - \
                                       (data[2] + self.gamma * self.Q_B(x_prime_A, self.maximizer.arg(self.Q_A, x_prime_A)))
                temporal_difference_B = self.Q_B(x_B[:5], x_B[5:]) - \
                                       (data[2] + self.gamma * self.Q_A(x_prime_B, self.maximizer.arg(self.Q_B, x_prime_B)))

                train_x_A.append(x_A)
                train_x_B.append(x_B)
                train_y_A.append(self.Q_A(x_A[:5], x_A[5:]) + self.learning_ratio * temporal_difference_A)
                train_y_B.append(self.Q_B(x_B[:5], x_B[5:]) + self.learning_ratio * temporal_difference_B)

                self.parameter_A = self.parameter_A - self.learning_ratio * temporal_difference_A * self.q_A_derivative(x_A)
                self.parameter_B = self.parameter_B - self.learning_ratio * temporal_difference_B * self.q_B_derivative(x_B)

            print('dataset formatted.')

            # Learning the q function
            print("Epoch" + str(i) + "...")
            self.Q_A.train(train_x_A, train_y_A)
            self.Q_B.train(train_x_B, train_y_B)
            print("function Q" + str(i) + " learnt.\n")

    def q_A_derivative(self, x):
        x_1, x_2 = x[:4] + [(x[4] + 0.0001)], x[:4] + [(x[4] - 0.0001)]
        y_1, y_2 = self.Q_A(x_1, x[5:]), self.Q_A(x_2, x[5:])
        return (y_2 - y_1) / 2 * 0.0001

    def q_B_derivative(self, x):
        x_1, x_2 = x[:4] + [(x[4] + 0.0001)], x[:4] + [(x[4] - 0.0001)]
        y_1, y_2 = self.Q_B(x_1, x[5:]), self.Q_B(x_2, x[5:])
        return (y_2 - y_1) / 2 * 0.0001

    def get_optimal_policy(self):
        p = self.parameter_A

        def foo(state, action):
            new_state = numpy.array([i for i in state] + [p])
            return self.Q_A(new_state, action)

        return FunctionPolicy(foo, self.maximizer)
