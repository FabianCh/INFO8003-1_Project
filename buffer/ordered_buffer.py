from buffer.buffer import Buffer


class OrderedBuffer(Buffer):
    """
        Return the sample from the first to the lst added
    """

    def __init__(self):
        self.samples = []

    def is_empty(self):
        return len(self.samples) == 0

    def add_sample(self, sample):
        if isinstance(sample, tuple):
            self.samples.append(sample)
        elif isinstance(sample, list):
            for i in sample:
                self.samples.append(i)

    def get_sample(self, n):
        res = []
        for i in range(n):
            res.append(self.samples[i % len(self.samples)])
        return res

    def clear(self):
        self.samples = []
