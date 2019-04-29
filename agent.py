from buffer.buffer import Buffer
from maximum_strategy import *
from catcher import ContinuousCatcher
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator

import imageio
import numpy


class Agent:
    def __init__(self, buffer, estimator):
        self.domain = ContinuousCatcher()
        self.gamma = self.domain.gamma()

        self.buffer = buffer
        self.estimator = estimator

        self.approximation_function_Qn = []

    def play(self, policy):
        """Play a game of catcher"""
        self.domain.reset()
        state = self.domain.observe()

        is_terminate = False
        number_of_action = 0
        cumulated_reward = 0
        while not is_terminate:
            action = policy(state)
            # print(action)
            state, reward, is_terminate = self.domain.step(action)
            number_of_action += 1
            cumulated_reward += reward
            if reward == 3:
                print ("Hit : ")
                print(state)
                print(is_terminate)

        # print("Game over, number of action =", number_of_action, "cumulated reward =", cumulated_reward)
        return cumulated_reward, number_of_action

    def reinitialize_buffer(self):
        """Reinitialize the buffer"""
        self.buffer.reset()

    def generate_one_step_transition(self, policy, n):
        """Complete the buffer with n one step transition with the given policy"""
        number_of_action = 0

        while number_of_action < n:
            self.domain.reset()
            initial_state = self.domain.observe()
            is_terminate = False

            while not is_terminate and number_of_action <= n:
                action = policy(initial_state)
                final_state, reward, is_terminate = self.domain.step(action)

                one_step_transition = (initial_state, action, reward, final_state)
                self.buffer.add_sample(one_step_transition)
                if reward == 3:
                    print("Hit : ")
                    print(initial_state)
                initial_state = final_state

                number_of_action += 1

    def expected_return(self, policy, n=100):
        """method to return the expected value with a policy in a domain"""
        cumulated_reward = 0
        for _ in range(n):
            cumulated_reward += self.play(policy)[0]
        cumulated_reward /= n
        return cumulated_reward

    def fitted_q_iteration(self, n):
        if self.buffer.is_empty():
            print("Empty Buffer")
            return

        for i in range(n):
            # collect the dataset
            print('Collecting dataset...')
            dataset = self.buffer.get_sample(1000)
            # print(dataset)
            print('dataset collected.\n')

            # Creation of the estimator
            if i >= len(self.approximation_function_Qn):
                print("Cretion of a new estimator")
                self.approximation_function_Qn.append(self.estimator())

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
                        self.gamma * maximum_uniform_sampling(self.approximation_function_Qn[i-1], x_prime, 7)
                    train_x.append(x)
                    train_y.append(y)
            print('dataset formatted.')

            # Learning the q function
            print("learning Q" + str(i) + "...")
            self.approximation_function_Qn[-1].train(train_x, train_y)

            print("function Q" + str(i) + " learnt.\n")

    def show(self, policy):
        self.domain.reset()
        state = self.domain.observe()

        window_width = self.domain.width
        window_height = self.domain.height

        y_bar = self.domain.bar.center[1]
        bar_width = self.domain.bar.size[0]
        bar_height = self.domain.bar.size[1]

        size_fruit = self.domain.fruit.size[0]

        def draw_bar(image, position):
            position = int(position)
            if 0 < position < window_width - bar_width//2:
                image[y_bar: y_bar + bar_height, position - bar_width // 2: position + bar_width // 2 + 1] = np.full((bar_height, bar_width), 1)

        def draw_fruit(image, x, y):
            x = int(x)
            y = int(y)
            if 0 < y < window_height - size_fruit:
                image[y: y + size_fruit, x: x + size_fruit] = np.full((size_fruit, size_fruit), 1)

        is_terminate = False
        number_of_action = 0
        cumulated_reward = 0
        with imageio.get_writer('animation.gif', mode='I', fps=10) as writer:
            while not is_terminate:
                img = numpy.zeros((window_height, window_width))
                state, reward, is_terminate = self.domain.step(policy(state))
                number_of_action += 1
                cumulated_reward += reward
                draw_bar(img, state[0])
                draw_fruit(img, state[2], state[3])
                writer.append_data(img)
                if reward == 3:
                    print("Hit : ")
                    print(state)

        return cumulated_reward, number_of_action
