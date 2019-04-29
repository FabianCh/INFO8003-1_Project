from policy.policy import Policy


class StaticPolicy(Policy):
    def __init__(self, action):
        self.action_value = action

    def __call__(self, state):
        return self.action_value
