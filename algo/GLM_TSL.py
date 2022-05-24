from sklearn.linear_model import LogisticRegression
from algo.zooming_ts_tuning import *


class GLM_TSL:
    def __init__(self, class_context):
        self.data = class_context
        self.T = self.data.T
        self.d = self.data.d

    def mu_dot(self, x):
        tmp = np.exp(-x)
        return tmp / ((1 + tmp) ** 2)

    def glmtsl_theoretical_explore(self, tau = 150, lamda = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)

        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        a = 0.5/0.1*math.sqrt(d*math.log(T/d)+ 2*math.log(T))*math.sqrt(0.25)
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

        if y[0] == y[1]:
            y[1] = 1 - y[0]

        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
            theta_bar = clf.coef_[0]
            H = np.identity(d) * lamda
            for l in range(t):
                tmp = self.mu_dot(X[l].dot(theta_bar))
                H += np.outer(X[l], X[l]) * tmp
            H_inv = np.linalg.inv(H)
            # for instable posterior solving due to unsuitable parameters, end early
            if np.isnan(H_inv).any() or np.isinf(H_inv).any() or np.isnan(theta_bar).any() or np.isinf(theta_bar).any():
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(theta_bar, a ** 2 * H_inv)
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret

    def glmtsl_auto(self, inte = [0,1], tau = 150, lamda = 1):
        T = self.T423
        d = self.data.d
        regret = np.zeros(T)

        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

        if y[0] == y[1]:
            y[1] = 1 - y[0]

        c = 13 / 2
        low = 0.7 * inte[0] + 0.3 * inte[1]
        up = inte[1] - low
        cen = (up - low) * np.random.random_sample(1) + low
        time = [1]
        rad = [math.sqrt(c * math.log(T))]
        sd = [math.sqrt(2 * c)]
        feature = self.data.fv[tau]
        K = len(feature)
        explore = trans(2, cen[0], 3)

        ucb_idx = [0] * K
        clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
        theta_bar = clf.coef_[0]
        H = np.identity(d) * lamda
        for l in range(tau):
            tmp = self.mu_dot(X[l].dot(theta_bar))
            H += np.outer(X[l], X[l]) * tmp
        H_inv = np.linalg.inv(H)
        # for instable posterior solving due to unsuitable parameters, end early
        theta = np.random.multivariate_normal(theta_bar, explore ** 2 * H_inv)
        for arm in range(K):
            ucb_idx[arm] = feature[arm].dot(theta)
        pull = np.argmax(ucb_idx)
        observe_r = self.data.random_sample(tau, pull)
        y = np.concatenate((y, [observe_r]), axis=0)
        X = np.concatenate((X, [feature[pull]]), axis=0)
        regret[tau] = regret[tau - 1] + self.data.optimal[tau] - self.data.reward[tau][pull]

        mu = [observe_r]

        for t in range((tau+1), T):
            feature = self.data.fv[t]
            K = len(feature)
            ind, cen, time, mu, rad, sd = auto_tuning(cen, time, rad, sd, c, T, mu, inte)
            ucb_idx = [0] * K
            clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
            theta_bar = clf.coef_[0]
            H = np.identity(d) * lamda
            for l in range(t):
                tmp = self.mu_dot(X[l].dot(theta_bar))
                H += np.outer(X[l], X[l]) * tmp
            H_inv = np.linalg.inv(H)
            # for instable posterior solving due to unsuitable parameters, end early
            if np.isnan(H_inv).any() or np.isinf(H_inv).any() or np.isnan(theta_bar).any() or np.isinf(theta_bar).any():
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(theta_bar, explore ** 2 * H_inv)
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            mu[ind] = (mu[ind] * (time[ind] - 1) + observe_r) / time[ind]
        return regret

    def glmtsl_op(self, explore_rates, tau = 150, lamda = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)

        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

        if y[0] == y[1]:
            y[1] = 1 - y[0]

        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)
        explore = explore_rates[index]

        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
            theta_bar = clf.coef_[0]
            H = np.identity(d) * lamda
            for l in range(t):
                tmp = self.mu_dot(X[l].dot(theta_bar))
                H += np.outer(X[l], X[l]) * tmp
            H_inv = np.linalg.inv(H)
            # for instable posterior solving due to unsuitable parameters, end early
            if np.isnan(H_inv).any() or np.isinf(H_inv).any() or np.isnan(theta_bar).any() or np.isinf(theta_bar).any():
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(theta_bar, explore ** 2 * H_inv)
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret

    def glmtsl_tl(self, explore_rates, tau = 150, lamda = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)

        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

        if y[0] == y[1]:
            y[1] = 1 - y[0]

        explore_lamda = explore_rates
        Kexp = len(explore_lamda)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / ((np.exp(1) - 1) * T)))
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore = explore_lamda[index]

        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(X, y)
            theta_bar = clf.coef_[0]
            H = np.identity(d) * lamda
            for l in range(t):
                tmp = self.mu_dot(X[l].dot(theta_bar))
                H += np.outer(X[l], X[l]) * tmp
            H_inv = np.linalg.inv(H)
            # for instable posterior solving due to unsuitable parameters, end early
            if np.isnan(H_inv).any() or np.isinf(H_inv).any() or np.isnan(theta_bar).any() or np.isinf(theta_bar).any():
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(theta_bar, explore ** 2 * H_inv)
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            logw, p, index = tl_auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_lamda[index]
        return regret