import numpy as np
import math


class context:
    def __init__(self, K, T, d, sigma, true_theta, theta_norm = -1, gen_type = 'uniform', model = 'linear', contextual = 'True', fv = None):
        self.ub = 1/math.sqrt(d)
        self.lb = -1/math.sqrt(d)
        if fv is None and contextual:
            if gen_type == 'uniform':
                fv = np.random.uniform(self.lb, self.ub, (T, K, d))
            else:
                fv = np.random.normal(size = (T, K, d))
                self.max_norm = float('-Inf')
                for t in range(T):
                    self.max_norm = max([np.linalg.norm(fea) for fea in fv[t]] + [self.max_norm])
                fv = fv/self.max_norm
        elif fv is None:
            if gen_type == 'uniform':
                fv = np.random.uniform(self.lb, self.ub, (K, d))
            else:
                fv = np.random.normal(size = (K, d))
                self.max_norm = max([np.linalg.norm(fea) for fea in fv])
                fv = fv/self.max_norm
        self.gen_type = gen_type
        self.K = K  
        self.d = d
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
        # self.max_norm = float('-Inf')
        self.sigma = sigma
        self.model = model
        self.contextual = contextual
        if theta_norm < 0:
            self.theta_norm = np.linalg.norm(self.theta)
        else:
            self.theta_norm = theta_norm

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def build_bandit(self):
        if self.contextual:
            if self.model == 'logistic':
                for t in range(self.T):
                    self.reward[t] = [self.logistic(self.fv[t][i].dot(self.theta)) for i in range(self.K)]
                    self.optimal[t] = max(self.reward[t])  # max reward
            else:
                for t in range(self.T):
                    self.reward[t] = [self.fv[t][i].dot(self.theta) for i in range(self.K)]
                    # make sure all rewards are within [0,1]
                    # self.reward[t] = (self.reward[t] + 1) / 2  # since x^T \theta lies in [-1,1]
                    self.optimal[t] = max(self.reward[t])
        else:
            if self.model == 'logistic':
                self.reward = [[self.logistic(self.fv[i].dot(self.theta)) for i in range(self.K)]] * self.T
                self.optimal = [max(self.reward)] * self.T  # max reward
            else:
                self.reward = [[self.fv[i].dot(self.theta) for i in range(self.K)]] * self.T
                # make sure all rewards are within [0,1]
                # self.reward[t] = (self.reward[t] + 1) / 2  # since x^T \theta lies in [-1,1]
                self.optimal = [max(self.reward)] * self.T

    def random_sample(self, t, i):
        if self.model == 'logistic':
            return np.random.binomial(1, self.reward[t][i])
        else:
            return np.random.normal(self.reward[t][i], self.sigma)


class movie:
    def __init__(self, K=100, T=10000, d=5, sigma=0.01, theta_norm = -1, model = 'linear', true_theta=None, fv=None):
        num_movie = len(fv)
        self.fv = np.zeros((T, K, d))
        # self.max_norm = float('-Inf')
        # un = []
        # for t in range(T):
        #     idx = np.random.choice(num_movie, K, replace=False)
        #     self.fv[t] = fv[idx, :]
        #     un = list(set().union(un, idx))
        # self.max_norm = np.max([np.linalg.norm(fv[i,:]) for i in un])
        # self.fv = self.fv / self.max_norm
        for t in range(T):
            idx = np.random.choice(num_movie, K, replace=False)
            max_norm = np.max([np.linalg.norm(fv[i,:]) for i in idx])
            self.fv[t] = fv[idx, :] / max_norm
        self.K = K
        self.d = d
        self.T = T
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
        self.sigma = sigma
        self.model = model
        if theta_norm < 0:
            self.theta_norm = np.linalg.norm(self.theta)
        else:
            self.theta_norm = theta_norm

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def build_bandit(self):
        # maxr = float('-Inf')
        # minr = float('Inf')
        if self.model == 'logistic':
            for t in range(self.T):
                self.reward[t] = [self.logistic(self.fv[t][i].dot(self.theta)) for i in range(self.K)]
                self.optimal[t] = max(self.reward[t])  # max reward
        else:
            for t in range(self.T):
                self.reward[t] = np.array([self.fv[t][i].dot(self.theta) for i in range(self.K)])
                # maxr = max(maxr, np.max(self.reward[t]))
                # minr = min(minr, np.min(self.reward[t]))
                # make sure rewards are within 0 to 1
                self.optimal[t] = max(self.reward[t])

    def random_sample(self, t, i):
        if self.model == 'logictic':
            return np.random.binomial(1, self.reward[t][i])
        else:
            return np.random.normal(self.reward[t][i], self.sigma)