import numpy as np
import random


class UniformSampler:
    def __init__(self, samples_number, arg_max=1.3):
        self.sample_number = samples_number
        self.arg_max = arg_max

    def value(self, function, state):
        samples = [random.uniform(-self.arg_max, 0) for _ in range(self.sample_number)] +\
                  [0] +\
                  [random.uniform(0, self.arg_max) for _ in range(self.sample_number)]
        maximum = function(state, [samples[0], 0])
        for arg in samples:
            value = function(state, [arg, 0])
            if value > maximum:
                maximum = value
        return maximum

    def arg(self, function, state):
        samples = [random.uniform(-self.arg_max, 0) for _ in range(self.sample_number)] + \
                  [0] + \
                  [random.uniform(0, self.arg_max) for _ in range(self.sample_number)]
        maximum_value = function(state, [samples[0], 0])
        maximum_argument = samples[0]
        for arg in samples:
            value = function(state, [arg, 0])
            if value > maximum_value:
                maximum_value = value
                maximum_argument = arg
        return maximum_argument, 0
