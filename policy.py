from maximum_strategy import *
import random

# state :  x_bar_center, bar_vel, x_fruit_center, fruit_center


class Policy:
    def __call__(self, *args, **kwargs):
        pass


class StaticPolicy(Policy):
    def __init__(self, action):
        self.action_value = action

    def __call__(self, state):
        return self.action_value


class RandomPolicy(Policy):
    def __call__(self, state):
        return random.uniform(-1.5, 1.5), 0


class OptimalPolicy(Policy):
    def __init__(self, function_qn_approximate):
        self.function_Qn_approximate = function_qn_approximate

    def __call__(self, state):
        return argument_maximum_uniform_sampling(self.function_Qn_approximate, state, 5)
