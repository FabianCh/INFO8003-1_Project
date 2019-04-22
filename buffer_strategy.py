import random
from maximum_strategy import *


def first_from_buffer(buffer, n):
    return buffer[:n]


def random_from_buffer(buffer, n):
    returned_buffer = []
    for _ in range(n):
        returned_buffer.append(random.choices(buffer))
    return returned_buffer


def one_step_transition_formatting_q0(one_step_transition):
    ost = one_step_transition
    x = [ost[0][0][0], ost[0][0][1], ost[0][0][2], ost[0][0][3], ost[0][1][0], ost[0][1][1]]
    y = ost[0][2]
    return x, y


def one_step_transition_formatting_qn(one_step_transition, function_qn, gamma):
    ost = one_step_transition
    x = [ost[0][0][0], ost[0][0][1], ost[0][0][2], ost[0][0][3], ost[0][1][0], ost[0][1][1]]
    y = ost[0][2] + gamma * maximum_uniform_sampling(function_qn, x[:4], 20)
    return x, y
