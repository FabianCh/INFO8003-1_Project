from buffer.buffer import Buffer
import random


class PriorityBuffer(Buffer):
    """
        Return the sample from the last returned to the most returned
    """

    def __init__(self):
        self.samples = []
        self.priority = []
        self.total_usage = 0

    def is_empty(self):
        return len(self.samples) == 0

    def add_sample(self, sample):
        if isinstance(sample, tuple):
            self.samples.append(sample)
            self.priority.append(0)
        elif isinstance(sample, list):
            for i in sample:
                self.samples.append(i)
                self.priority.append(0)

    def get_sample(self, n):
        self.total_usage += n
        hold = random.choices(range(len(self.priority)),
                              [1 - (self.priority[i]/self.total_usage) for i in range(len(self.priority))],
                              k=n)
        res = []
        for i in hold:
            self.priority[i] += 1
            res.append(self.samples[i])
        return res

    def reset(self):
        self.priority = [0 for _ in range(len(self.samples))]

    def clear(self):
        self.samples = []
        self.priority = []
        self.total_usage = 0
