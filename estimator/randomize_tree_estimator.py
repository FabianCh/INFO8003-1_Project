from estimator.estimator import Estimator
from sklearn.ensemble.forest import ExtraTreesRegressor
import numpy as np


class ExtremelyRandomizeTreeEstimator(Estimator):

    def __init__(self):
        self.estimator = ExtraTreesRegressor(n_estimators=20)

    def __call__(self, state, action):
        x = np.array([state[0], state[1], state[2], state[3], action[0], action[1]]).reshape(1, -1)
        return self.estimator.predict(x)[0]

    def train(self, train_in, train_out):
        train_in_formatted = []
        for sample in train_in:
            train_in_formatted.append([sample[0][0], sample[0][1], sample[0][2], sample[0][3],
                                       sample[1][0], sample[1][1]])
        train_in_formatted = np.array(train_in_formatted)
        self.estimator.fit(train_in_formatted, train_out)
