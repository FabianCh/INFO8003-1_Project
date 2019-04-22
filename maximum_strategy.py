import numpy as np


def maximum_uniform_sampling(function_qn, state,  number_of_sample):
    sampling = np.linspace(-1.344, 1.344, number_of_sample)
    maximum = function_qn(state, [-1.344, 0])
    for i in sampling:
        value = function_qn(state, [i, 0])
        if value > maximum:
            maximum = value
    return maximum


def argument_maximum_uniform_sampling(function_qn, state,  number_of_sample):
    sampling = np.linspace(-1.344, 1.344, number_of_sample)
    argument_maximum = -1.344
    maximum = function_qn(state, [-1.344, 0])
    for i in sampling:
        value = function_qn(state, [i, 0])
        if value > maximum:
            argument_maximum = i
            maximum = value
    return argument_maximum, 0
