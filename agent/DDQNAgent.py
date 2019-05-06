from agent.DQNAgent import DQNAgent


class DDQNAgent(DQNAgent):
    def train(self, epoch=1, mini_batch=30):
        dataset = self.buffer.get_sample(mini_batch)

        train_x = []
        train_y = []

        for state, action, reward, next_state in dataset:
            x = [state[0], state[1], state[2], state[3], action[0], action[1]]
            x_prime = [next_state[0], next_state[1], next_state[2], next_state[3]]
            y = reward + \
                self.gamma * self.target_Q(x_prime, self.maximizer.arg(self.Q, x_prime))
            train_x.append(x)
            train_y.append(y)
        self.Q.train(train_x, train_y)
