from catcher import ContinuousCatcher
from estimator.randomize_tree_estimator import ExtremelyRandomizeTreeEstimator
from buffer_strategy import *


class Agent:
    def __init__(self):
        self.approximation_function_Qn = []
        self.buffer = []
        self.domain = ContinuousCatcher()
        self.gamma = self.domain.gamma()

    def play(self, policy):
        self.domain.reset()
        state = self.domain.observe()

        is_terminate = False
        number_of_action = 0
        cumulated_reward = 0
        while not is_terminate:
            state, reward, is_terminate = self.domain.step(policy(state))
            number_of_action += 1
            cumulated_reward += reward

        # print("Game over, number of action =", number_of_action, "cumulated reward =", cumulated_reward)
        return cumulated_reward, number_of_action

    def reinitialize_buffer(self):
        """Reinitialize the buffer"""
        self.buffer = []

    def generate_one_step_transition(self, policy, n):
        """Complete the buffer with n one step transition with the given policy"""
        number_of_action = 0

        while True:
            self.domain.reset()
            state = self.domain.observe()

            is_terminate = False
            while not is_terminate:
                action = policy(state)
                one_step_transition = [state, action]

                state, reward, is_terminate = self.domain.step(action)

                one_step_transition.append(reward)
                one_step_transition.append(state)
                self.buffer.append(one_step_transition)
                number_of_action += 1
                if number_of_action >= n:
                    return None

    def expected_return(self, policy, n=100):
        """method to return the expected value with a policy in a domain"""
        cumulated_reward = 0
        for _ in range(n):
            cumulated_reward += self.play(policy)[0]
        cumulated_reward /= n
        return cumulated_reward

    def fitted_q_iteration(self, n):
        # collect the dataset
        print('Collecting dataset...')
        dataset = random_from_buffer(self.buffer, 500)
        print('dataset collected.\n')

        i = 0
        while i <= n:
            # Creation of the estimator
            self.approximation_function_Qn.append(ExtremelyRandomizeTreeEstimator())

            # formatting the dataset
            print('formatting the dataset...')
            train_x = []
            train_y = []

            if i == 0:
                for data in dataset:
                    x, y = one_step_transition_formatting_q0(data)
                    train_x.append(x)
                    train_y.append(y)
            else:
                for data in dataset:
                    x, y = one_step_transition_formatting_qn(data, self.approximation_function_Qn[i-1], self.gamma)
                    train_x.append(x)
                    train_y.append(y)
            print('dataset formatted.')

            # Learning the q function
            print("learning Q" + str(i) + "...")
            self.approximation_function_Qn[-1].train(train_x, train_y)

            print("function Q" + str(i) + " learnt.\n")
            i += 1
