class Policy:
    def __call__(self, state):
        """
            Return the action given a state
        """
        pass


# class OptimalPolicy(Policy):
#     def __init__(self, function_qn_approximate):
#         self.function_Qn_approximate = function_qn_approximate
#
#     def __call__(self, state):
#         return argument_maximum_uniform_sampling(self.function_Qn_approximate, state, 7)
