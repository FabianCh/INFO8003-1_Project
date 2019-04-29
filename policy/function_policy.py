from policy.policy import Policy


class FunctionPolicy(Policy):
    def __init__(self, function, maximizer):
        self.function = function
        self.maximizer = maximizer

    def __call__(self, state):
        return self.maximizer.arg(self.function, state)
