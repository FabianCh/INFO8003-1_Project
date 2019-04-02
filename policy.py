# state :  x_bar_center, bar_vel, x_fruit_center, fruit_center


class Policy:
    def __call__(self, *args, **kwargs):
        pass


class StaticPolicy(Policy):
    def __init__(self, action):
        self.action_value = action

    def __call__(self, state):
        return self.action_value
