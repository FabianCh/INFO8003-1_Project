from policy.policy import Policy

import random


class GreedyPolicy(Policy):

    def __init__(self, function, maximizer, epsilon):
        assert 0 <= epsilon <= 1, "epsilon must be between [0,1]"
        self.function = function
        self.maximizer = maximizer
        self.epsilon = epsilon

    def __call__(self, state):
        r = random.uniform(0, 1)
        if r < self.epsilon:
            return random.uniform(-1.5, 1.5), 0
        else:
            return self.maximizer.arg(self.function, state)
