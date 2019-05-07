from catcher import ContinuousCatcher
from policy.random_policy import RandomPolicy
from maximizer.static_sampler import StaticSampler
from buffer.random_buffer import RandomBuffer

import imageio
import numpy


class Agent:
    def __init__(self, estimator, buffer=RandomBuffer(), maximizer=StaticSampler(3), policy=RandomPolicy()):
        self.domain = ContinuousCatcher()
        self.gamma = self.domain.gamma()

        self.estimator = estimator
        self.buffer = buffer
        self.maximizer = maximizer

        self.policy = policy

    def reinitialize_buffer(self):
        """
            Reinitialize the buffer
        """
        self.buffer.reset()

    def reinitialize_domain(self):
        """
            Reinitialize the domain
        """
        self.domain.reset()

    def set_policy(self, policy):
        """
            Set the policy of the agent
        """
        self.policy = policy

    def play(self, return_hits=False, verbose=True):
        """
            Play a game of catcher
        """

        if verbose:
            print("Game started")

        domain = ContinuousCatcher()
        state, is_terminate = domain.observe(), False
        cumulated_reward, hits = 0, 0

        while not is_terminate:
            action = self.policy(state)
            state, reward, is_terminate = domain.step(action)

            cumulated_reward += reward
            if reward == 3:
                hits += 1
                if verbose:
                    print("    Hit : " + str(state))

        if verbose:
            print("Game ended :")
            print("  Hits = ", hits)
            print("  Cumulated reward = ", cumulated_reward)

        if return_hits:
            return cumulated_reward, hits
        return cumulated_reward

    def show(self, animation_title=None, verbose=False):
        """
            Play a game an create an animation
        """

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

        if verbose:
            print("Game started")

        domain = ContinuousCatcher()
        state, is_terminate = domain.observe(), False
        cumulated_reward, hits = 0, 0

        with imageio.get_writer(animation_title + ".gif", mode='I', fps=10) as writer:
            while not is_terminate:
                action = self.policy(state)
                state, reward, is_terminate = domain.step(action)

                cumulated_reward += reward
                if reward == 3:
                    hits += 1
                    if verbose:
                        print("    Hit : " + str(state))
                try:
                    img = numpy.zeros((window_height, window_width))
                    draw_bar(img, state[0])
                    draw_fruit(img, state[2], state[3])
                    writer.append_data(img)
                except:
                    pass
        if verbose:
            print("Game ended :")
            print("  Hits = ", hits)
            print("  Cumulated reward = ", cumulated_reward)

    def generate_one_step_transition(self, n=None):
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

            action = self.policy(initial_state)
            final_state, reward, is_terminate = self.domain.step(action)

            one_step_transition = (initial_state, action, reward, final_state)
            self.buffer.add_sample(one_step_transition)

            initial_state = final_state
            action_number += 1

    def expected_return(self, n=100, verbose=False):
        """
            Return the expected value with a policy in a domain
        """
        cumulated_reward = 0
        if verbose:
            print("Expected return started")
        for _ in range(n):
            cumulated_reward += self.play(verbose=False)[0]

        mean_reward = cumulated_reward / n

        if verbose:
            print("Extected return : " + str(mean_reward))

        return mean_reward

    def expected_return_and_hit(self, n=100, verbose=False):
        """
            Return the expected value with a policy in a domain
        """
        cumulated_reward, cumulated_hits = 0, 0

        if verbose:
            print("Expected return started")

        for _ in range(n):
            reward, hits = self.play(return_hits=True, verbose=False)
            cumulated_reward += reward
            cumulated_hits += hits

        mean_reward = cumulated_reward / n
        mean_hits = cumulated_hits / n

        if verbose:
            print("\n Extected return : " + str(mean_reward))
            print("\n Extected hits : " + str(mean_hits))

        return mean_reward, mean_hits

    def play_and_train(self, n=None):
        """
            Play n action and train the agent according to its training policy
        """
        pass

    def train(self, depth=None, dataset_size=None):
        """
            Train the agent on a data set
        """
        pass

    def get_optimal_policy(self):
        """
            Get the optimal policy of an agent
        """
        pass

    def get_greedy_policy(self):
        """
            Get a greedy policy for the agent
        """
        pass
