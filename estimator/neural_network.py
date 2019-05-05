import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class NeuralNetworkEstimator:

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(6,), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='rmsprop', loss='mse')

    def __call__(self, state, action):
        x = np.array([state[i] for i in range(len(state))] + [action[0], action[1]]).reshape(1, -1)
        return self.model.predict(x)[0]

    def train(self, train_in, train_out):

        train_in = np.array(train_in)
        train_out = np.array(train_out)
        self.model.fit(train_in, train_out, epochs=10, batch_size=5, verbose=0)

    def copy(self, neural_network):
        self.model.set_weights(neural_network.model.get_weights())



