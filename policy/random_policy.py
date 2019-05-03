from policy.policy import Policy

import random


class RandomPolicy(Policy):
    def __call__(self, state):
        return random.uniform(-1.5, 1.5), 0
