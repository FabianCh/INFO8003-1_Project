from catcher import ContinuousCatcher
from policy.function_policy import FunctionPolicy

import imageio
import numpy


class Agent:
    def __init__(self, buffer, estimator, maximizer):
        self.domain = ContinuousCatcher()
        self.gamma = self.domain.gamma()

        self.buffer = buffer
        self.estimator = estimator
        self.maximizer = maximizer

    def reinitialize_buffer(self):
        """
            Reinitialize the buffer
        """
        self.buffer.reset()

    def play(self, policy):
        """
            Play a game of catcher
        """
        print("Game started")
        self.domain.reset()
        state = self.domain.observe()

        is_terminate = False
        number_of_action = 0
        cumulated_reward = 0
        while not is_terminate:
            action = policy(state)
            state, reward, is_terminate = self.domain.step(action)
            number_of_action += 1
            cumulated_reward += reward
            if reward == 3:
                print("    Hit : " + str(state))

        print("Game ended :")
        print("  number of action = ", number_of_action)
        print("  cumulated reward = ", cumulated_reward)
        return cumulated_reward, number_of_action

    def show(self, policy, animation_title=None):
        if animation_title is None:
            animation_title = "animation"
        window_width, window_height = self.domain.width, self.domain.height
        bar_width, bar_height = self.domain.bar.size
        size_fruit = self.domain.fruit.size[0]
        y_bar = self.domain.bar.center[1]

        def draw_bar(image, position):
            position = int(position)
            if 0 < position < window_width - bar_width//2:
                image[y_bar: y_bar + bar_height, position - bar_width // 2: position + bar_width // 2 + 1] =\
                    numpy.full((bar_height, bar_width), 1)

        def draw_fruit(image, x, y):
            x = int(x)
            y = int(y)
            if 0 < y < window_height - size_fruit:
                image[y: y + size_fruit, x: x + size_fruit] = numpy.full((size_fruit, size_fruit), 1)

        print("Game started")
        self.domain.reset()
        state = self.domain.observe()
        is_terminate = False
        number_of_action, cumulated_reward = 0, 0

        with imageio.get_writer(animation_title + ".gif", mode='I', fps=10) as writer:
            while not is_terminate:
                action = policy(state)
                state, reward, is_terminate = self.domain.step(action)
                number_of_action += 1
                cumulated_reward += reward
                if reward == 3:
                    print("    Hit : " + str(state))

                img = numpy.zeros((window_height, window_width))
                draw_bar(img, state[0])
                draw_fruit(img, state[2], state[3])
                writer.append_data(img)

        print("Game ended :")
        print("  number of action = ", number_of_action)
        print("  cumulated reward = ", cumulated_reward)
        return cumulated_reward, number_of_action

    def generate_one_step_transition(self, policy, n=None):
        """
            Complete the buffer with n one step transition with the given policy
            If no n is given add a game to the buffer
        """
        self.domain.reset()
        initial_state, is_terminate = self.domain.observe(), False
        action_number = 0
        if n is None:
            def test(terminated, n, action_number):
                return not terminated
        else:
            def test(terminated, n, action_number):
                return action_number < n
        while test(is_terminate, n, action_number):
            if is_terminate:
                self.domain.reset()
                initial_state = self.domain.observe()

            action = policy(initial_state)
            final_state, reward, is_terminate = self.domain.step(action)

            one_step_transition = (initial_state, action, reward, final_state)
            self.buffer.add_sample(one_step_transition)

            initial_state = final_state
            action_number += 1

    def expected_return(self, policy, n=100):
        """
            Return the expected value with a policy in a domain
        """
        cumulated_reward = 0
        for _ in range(n):
            cumulated_reward += self.play(policy)[0]

        res = cumulated_reward / n
        print("\n Extected return : " + str(res))
        return res

    def train(self, depth=None, dataset_size=None):
        pass

    def get_optimal_policy(self):
        pass