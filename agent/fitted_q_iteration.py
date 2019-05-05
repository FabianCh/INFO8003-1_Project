from agent.agent import Agent
from policy.function_policy import FunctionPolicy


class FittedQIteration(Agent):
    def __init__(self, buffer, estimator, maximizer):
        super(FittedQIteration, self).__init__(buffer, estimator, maximizer)
        self.Qn = []

    def train(self, depth=100, dataset_size=100):
        if self.buffer.is_empty():
            print("Empty Buffer")
            return

        for i in range(depth):
            # collect the dataset
            print("Collecting dataset...")
            dataset = self.buffer.get_sample(dataset_size)
            print('Dataset collected.\n')

            # Creation of the estimator
            if i >= len(self.Qn):
                print("Cretion of a new estimator")
                self.Qn.append(self.estimator())

            # formatting the dataset
            print('formatting the dataset...')
            train_x = []
            train_y = []

            if i == 0:
                for data in dataset:
                    x = [data[0][0], data[0][1], data[0][2], data[0][3], data[1][0], data[1][1]]
                    y = data[2]
                    train_x.append(x)
                    train_y.append(y)
            else:
                for data in dataset:
                    x = [data[0][0], data[0][1], data[0][2], data[0][3], data[1][0], data[1][1]]
                    x_prime = [data[3][0], data[3][1], data[3][2], data[3][3]]
                    y = data[2] +\
                        self.gamma * self.maximizer.value(self.Qn[i - 1], x_prime)
                    train_x.append(x)
                    train_y.append(y)

            print('dataset formatted.')

            # Learning the q function
            print("learning Q" + str(i) + "...")
            self.Qn[-1].train(train_x, train_y)
            print("function Q" + str(i) + " learnt.\n")

    def get_optimal_policy(self):
        return FunctionPolicy(self.Qn[-1], self.maximizer)
