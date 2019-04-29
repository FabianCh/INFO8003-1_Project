import random


def first_from_buffer(buffer, n):
    return buffer[:n]


def random_from_buffer(buffer, n):
    returned_buffer = []
    for _ in range(n):
        returned_buffer.append(random.choices(buffer))
    return returned_buffer