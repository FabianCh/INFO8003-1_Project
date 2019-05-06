class Buffer:

    def is_empty(self):
        """
            Check if the buffer is empty
        """
        pass

    def add_sample(self, sample):
        """
            Add a sample to the history
            If sample is a list add each element as 1 sample
        """
        pass

    def get_sample(self, n):
        """
            Return n sample from the buffer according to the selection policy
        """
        pass

    def reset(self):
        """
            Reset the selection policy of the buffer
        """
        pass

    def clear(self):
        """
            Clear the buffer
        """
        pass
