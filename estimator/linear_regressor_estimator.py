from estimator.estimator import Estimator

import numpy as np
import sklearn.linear_model as skl_lm
import sklearn.preprocessing as skl_pp


class LinearRegressorEstimator(Estimator):

    def __init__(self):
        self.estimator = skl_lm.LinearRegression()
        self.scaler = skl_pp.StandardScaler()

    def __call__(self, state, action):
        x = np.array([state[i] for i in range(len(state))] + [action[0], action[1]]).reshape(1, -1)
        return self.estimator.predict(x)[0]

    def train(self, train_in, train_out):

        train_in_formatted = np.array(train_in)
        self.estimator.fit(train_in_formatted, train_out)
