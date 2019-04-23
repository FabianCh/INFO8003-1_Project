import random
from maximum_strategy import *


def first_from_buffer(buffer, n):
    return buffer[:n]


def random_from_buffer(buffer, n):
    returned_buffer = []
    for _ in range(n):
        returned_buffer.append(random.choices(buffer))
    return returned_buffer


def one_step_transition_formatting_q0(ost):
    x = [ost[0][0], ost[0][1], ost[0][2], ost[0][3], ost[1][0], ost[1][1]]
    y = ost[0][2]
    return x, y


def one_step_transition_formatting_qn(ost, function_qn, gamma):
    x = [ost[0][0], ost[0][1], ost[0][2], ost[0][3], ost[1][0], ost[1][1]]
    x_prime = [ost[3][0], ost[3][1], ost[3][2], ost[3][3]]
    y = ost[0][2] + gamma * maximum_uniform_sampling(function_qn, x_prime, 5)
    return x, y
