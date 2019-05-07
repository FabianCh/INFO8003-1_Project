from maximizer.maximizer import Maximizer

import numpy as np


class StaticSampler(Maximizer):
    def __init__(self, samples_number, arg_max=1.3):
        self.sample_number = samples_number
        self.arg_max = arg_max
        self.samples = np.linspace(-self.arg_max, self.arg_max, self.sample_number)

    def value(self, function, state):
        maximum = function(state, [self. samples[0], 0])
        for arg in self.samples:
            value = function(state, [arg, 0])
            if value > maximum:
                maximum = value
        return maximum

    def arg(self, function, state):
        maximum_value = function(state, [self.samples[0], 0])
        maximum_argument = self.samples[0]
        for arg in self.samples:
            value = function(state, [arg, 0])
            if value > maximum_value:
                maximum_value = value
                maximum_argument = arg
        return maximum_argument, 0
