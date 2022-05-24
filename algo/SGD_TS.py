from sklearn.linear_model import LogisticRegression
from algo.zooming_ts_tuning import *
import itertools as it


class SGD_TS:
    def __init__(self, class_context):
        self.data = class_context
        self.T = self.data.T
        self.d = self.data.d

    def grad(self, x, y, theta, lamda=0):
        return x * (-y + 1 / (1 + np.exp(-x.dot(theta)))) + 2 * lamda * theta

    def sgdts_theoretical_explore(self, C, eta0):  # paras should be a dictionary of hyper-paras to be tuned
        # eta0s, g1s, g2s, lamdas):
        T = self.T
        d = self.d
        tau = C * max(math.log(T), d)

        regret = np.zeros(self.T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
        if y[0] == y[1]:
            y[1] = 1 - y[0]

        # initialize for exp3
        # order_prior = {'eta0': 1,
        #                'alpha1': 2,
        #                'alpha2': 3,
        #                }
        # keys = list(paras.keys())
        # keys = sorted(keys, key=lambda kv: order_prior[kv])
        # combinations = list(it.product(*(paras[kv] for kv in keys)))

        clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
        theta_hat = clf.coef_[0]
        grad = np.zeros(d)
        theta_tilde = np.zeros(d)
        theta_tilde[:] = theta_hat[:]
        theta_bar = np.zeros(d)

        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            if t % tau == 0:
                j = t // tau

                # theoretical value same as UCB-GLM
                g1 = self.data.sigma * math.sqrt(d / 2 * math.log(1 + 2 * j * tau / d) + 2 * math.log(T))
                # theoretical value defined in sgdts paper
                g2 = tau / eta0 * math.sqrt(1 + math.log(j))

                cov = (2 * g1 ** 2 + 2 * g2 ** 2) * np.identity(d) / j
                eta = eta0 / j
                theta_tilde -= eta * grad
                distance = np.linalg.norm(theta_tilde - theta_hat)
                if distance > 2:
                    theta_tilde = theta_hat + 2 * (theta_tilde - theta_hat) / distance
                grad = np.zeros(d)
                reward_exp3 = 0
                theta_bar = (theta_bar * (j - 1) + theta_tilde) / j
                theta_ts = np.random.multivariate_normal(theta_bar, cov)

            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)
            grad += self.grad(feature[pull], observe_r, theta_tilde, 0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

        return regret



    def sgdts_auto(self, C, inte = [[0,1],[0,1]]):  # paras should be a dictionary of hyper-paras to be tuned
        # eta0s, g1s, g2s, lamdas):
        T = self.T
        d = self.d
        tau = C * max(math.log(T), d)

        regret = np.zeros(self.T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
        if y[0] == y[1]:
            y[1] = 1 - y[0]

        # initialize for exp3
        # order_prior = {'eta0': 1,
        #                'alpha1': 2,
        #                'alpha2': 3,
        #                }
        # keys = list(paras.keys())
        # keys = sorted(keys, key=lambda kv: order_prior[kv])
        # combinations = list(it.product(*(paras[kv] for kv in keys)))

        clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
        theta_hat = clf.coef_[0]
        grad = np.zeros(d)
        theta_tilde = np.zeros(d)
        theta_tilde[:] = theta_hat[:]
        theta_bar = np.zeros(d)

        c = 13 / 8
        low = 0.7 * inte[0] + 0.3 * inte[1]
        up = inte[1] - low
        cen = [(up - low) * np.random.random_sample(2) + low]
        time = [1]
        rad = [math.sqrt(c * math.log(T))]
        sd = [math.sqrt(2 * c)]
        explore = cen[0]
        i = 0
        reward_tune = 0
        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            if t % tau == 0:
                if i > 0:
                    ind, cen, time, mu, rad, sd = auto_tuning(cen, time, rad, sd, c, T, mu, inte)
                    explore = cen[ind]
                j = t // tau
                eta0 = explore[1]
                # theoretical value same as UCB-GLM
                # g1 = self.data.sigma * math.sqrt(d / 2 * math.log(1 + 2 * j * tau / d) + 2 * math.log(T))
                # # theoretical value defined in sgdts paper
                # g2 = tau / eta0 * math.sqrt(1 + math.log(j))

                cov = (2 * explore[0]**2) * np.identity(d) / j
                eta = eta0 / j
                theta_tilde -= eta * grad
                distance = np.linalg.norm(theta_tilde - theta_hat)
                if distance > 2:
                    theta_tilde = theta_hat + 2 * (theta_tilde - theta_hat) / distance
                grad = np.zeros(d)
                theta_bar = (theta_bar * (j - 1) + theta_tilde) / j
                theta_ts = np.random.multivariate_normal(theta_bar, cov)
                if i == 0:
                    mu = [reward_tune]
                else:
                    mu[ind] = (mu[ind] * (time[ind] - 1) + reward_tune) / time[ind]
                reward_tune = 0
                i += 1

            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)
            grad += self.grad(feature[pull], observe_r, theta_tilde, 0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            reward_tune += observe_r

        return regret


    def sgdts_op(self, C, explore_rates):  # paras should be a dictionary of hyper-paras to be tuned
        # eta0s, g1s, g2s, lamdas):
        T = self.T
        d = self.data.d
        tau = C * max(math.log(T), d)

        regret = np.zeros(self.T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
        if y[0] == y[1]:
            y[1] = 1 - y[0]


        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)

        clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
        theta_hat = clf.coef_[0]
        grad = np.zeros(d)
        theta_tilde = np.zeros(d)
        theta_tilde[:] = theta_hat[:]
        theta_bar = np.zeros(d)

        reward_exp3 = 0
        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            if t % tau == 0:
                j = t // tau
                # update explore rates by auto_tuning
                s, f, index = op_tuning(s, f, reward_exp3, index)
                explore = explore_rates[index]
                # theoretical value same as UCB-GLM
                g1 = self.data.sigma * math.sqrt(d / 2 * math.log(1 + 2 * j * tau / d) + 2 * math.log(T))

                g2 = tau / explore * math.sqrt(1 + math.log(j))

                cov = (2 * g1 ** 2 + 2 * g2 ** 2) * np.identity(d) / j
                eta = explore / j
                theta_tilde -= eta * grad
                distance = np.linalg.norm(theta_tilde - theta_hat)
                if distance > 2:
                    theta_tilde = theta_hat + 2 * (theta_tilde - theta_hat) / distance
                grad = np.zeros(d)
                reward_exp3 = 0
                theta_bar = (theta_bar * (j - 1) + theta_tilde) / j
                theta_ts = np.random.multivariate_normal(theta_bar, cov)

            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)

            grad += self.grad(feature[pull], observe_r, theta_tilde, 0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            reward_exp3 += observe_r
        return regret

    def sgdts_tl(self, C, paras):  # paras should be a dictionary of hyper-paras to be tuned
        # eta0s, g1s, g2s, lamdas):
        T = self.T
        d = self.data.d
        tau = C * max(math.log(T), d)

        regret = np.zeros(self.T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
        if y[0] == y[1]:
            y[1] = 1 - y[0]

        # initialize for exp3
        order_prior = {'eta0': 1,
                       'alpha': 2,
                       }
        keys = list(paras.keys())
        keys = sorted(keys, key=lambda kv: order_prior[kv])
        combinations = list(it.product(*(paras[kv] for kv in keys)))

        Kexp = len(combinations)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / ((np.exp(1) - 1) * T)))
        # random initial explore rate
        index = np.random.choice(Kexp)
        selected_paras = {}

        clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
        theta_hat = clf.coef_[0]
        grad = np.zeros(d)
        theta_tilde = np.zeros(d)
        theta_tilde[:] = theta_hat[:]
        theta_bar = np.zeros(d)

        reward_exp3 = 0
        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            if t % tau == 0:
                j = t // tau
                # update explore rates by auto_tuning
                logw, p, index = tl_auto_tuning(logw, p, reward_exp3 / tau, index, gamma)
                for i in range(len(keys)):
                    selected_paras[keys[i]] = list(combinations[index])[i]

                # use the hyper-para selected by exp3
                if 'eta0' in paras.keys(): eta0 = selected_paras['eta0']
                if 'alpha' in paras.keys():
                    g1 = selected_paras['alpha']**2
                else:
                    # theoretical value same as UCB-GLM
                    g1 = (self.data.sigma * math.sqrt(d / 2 * math.log(1 + 2 * j * tau / d) + 2 * math.log(T)))**2 + \
                         (tau / eta0 * math.sqrt(1 + math.log(j)))**2

                cov = (2 * g1) * np.identity(d) / j
                eta = eta0 / j
                theta_tilde -= eta * grad
                distance = np.linalg.norm(theta_tilde - theta_hat)
                if distance > 2:
                    theta_tilde = theta_hat + 2 * (theta_tilde - theta_hat) / distance
                grad = np.zeros(d)
                reward_exp3 = 0
                theta_bar = (theta_bar * (j - 1) + theta_tilde) / j
                theta_ts = np.random.multivariate_normal(theta_bar, cov)

            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)

            grad += self.grad(feature[pull], observe_r, theta_tilde, 0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            reward_exp3 += observe_r

        return regret
