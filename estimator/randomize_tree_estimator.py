from estimator.estimator import Estimator
from sklearn.ensemble.forest import ExtraTreesRegressor
import numpy as np


class ExtremelyRandomizeTreeEstimator(Estimator):

    def __init__(self):
        self.estimator = ExtraTreesRegressor(n_estimators=30)
        self.initialized = False

    def __call__(self, state, action):
        if self.initialized:
            x = np.array([state[i] for i in range(len(state))] + [action[0], action[1]]).reshape(1, -1)
            return self.estimator.predict(x)[0]
        else:
            return 0

    def train(self, train_in, train_out):
        self.initialized = True
        train_in_formatted = train_in
        train_in_formatted = np.array(train_in_formatted)
        self.estimator.fit(train_in_formatted, train_out)