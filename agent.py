from catcher import ContinuousCatcher


class Agent:
    def __init__(self):
        self.approximation_function_Qn = []
        self.buffer = []
        self.domain = ContinuousCatcher()

    def play(self, policy):
        self.domain.reset()
        state = self.domain.observe()

        is_terminate = False
        number_of_action = 0
        cumulated_reward = 0
        while not is_terminate:
            state, reward, is_terminate = self.domain.step(policy(state))
            number_of_action += 1
            cumulated_reward += reward

        # print("Game over, number of action =", number_of_action, "cumulated reward =", cumulated_reward)
        return cumulated_reward, number_of_action

    def expected_return(self, policy, n=100):
        """method to return the expected value with a policy in a domain"""
        cumulated_reward = 0
        for _ in range(n):
            cumulated_reward += self.play(policy)[0]
        cumulated_reward /= n
        return cumulated_reward

    def fitted_q_iteration(self, n):
        # TODO The function to learn estimator of Qn
        pass
