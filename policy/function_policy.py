from policy.policy import Policy


class FunctionPolicy(Policy):
    """
            Retunn the action that maximize the function according to the maximizer
    """
    def __init__(self, function, maximizer):
        self.function = function
        self.maximizer = maximizer

    def __call__(self, state):
        return self.maximizer.arg(self.function, state)
