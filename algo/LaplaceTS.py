from algo.zooming_ts_tuning import *


class Laplace_TS:
    def __init__(self, class_context):
        self.data = class_context
        self.T = self.data.T
        self.d = self.data.d


    def grad(self, w, q, m, X, y):
        d = self.d
        g = np.zeros(d)
        for i in range(len(X)):
            g += (w - m) * q - y[i] * X[i] / (1 + np.exp(y[i] * X[i].dot(w)))
        return g

    def optimize(self, w, m, q, X, y, eta, max_ite):
        d = self.d
        w = np.random.uniform(-1, 1, d) + m
        for i in range(max_ite):
            grad = self.grad(w, q, m, X, y)
            grad_norm = np.linalg.norm(grad)
            if grad_norm <= 10 ** (-4):
                break
            if i % 100 == 0:
                eta /= 2
            w -= eta * grad
        return w

    def laplacets_theoretical_explore(self, lamda=1, eta0=0.1, max_ite=1000):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        m = np.zeros(d)
        q = np.ones(d) * lamda
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            if np.isnan(m).any() or np.isnan(q).any() or np.isinf(m).any() or np.isinf(q).any():
                # print('inf or nan encountered in posterior, will change to another step size to continue grid search')
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(m, np.diag(1 / q))
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.array([2 * observe_r - 1])
            X = np.array([feature[pull]])
            w = self.optimize(theta, m, q, X, y, eta0, max_ite)
            m[:] = w[:]
            p = self.data.logistic(feature[pull].dot(w))
            q += p * (1 - p) * feature[pull] ** 2
        return regret

    def laplacets_auto(self, exp_time = 0, inte = [0,1], lamda = 1, max_ite = 1000):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        m = np.zeros(d)
        q = np.ones(d) * lamda

        for t in range(exp_time):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

        c = 13 / 8
        low = 0.7 * inte[0] + 0.3 * inte[1]
        up = inte[1] - low
        cen = (up - low) * np.random.random_sample(1) + low
        time = [1]
        rad = [math.sqrt(c * math.log(T))]
        sd = [math.sqrt(2 * c)]
        feature = self.data.fv[exp_time]
        K = len(feature)
        explore = trans(3,cen[0],3)

        ts_idx = [0] * K

        theta = np.random.multivariate_normal(m, np.diag(1 / q))
        for arm in range(K):
            ts_idx[arm] = feature[arm].dot(theta)
        pull = np.argmax(ts_idx)
        observe_r = self.data.random_sample(t, pull)
        regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
        y = np.array([2 * observe_r - 1])
        X = np.array([feature[pull]])
        w = self.optimize(theta, m, q, X, y, explore, max_ite)
        m[:] = w[:]
        p = self.data.logistic(feature[pull].dot(w))
        q += p * (1 - p) * feature[pull] ** 2

        mu = [observe_r]


        for t in range((exp_time+1),T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            ind, cen, time, mu, rad, sd = auto_tuning(cen, time, rad, sd, c, T, mu, inte)
            explore = trans(3,cen[ind],3)
            if np.isnan(m).any() or np.isnan(q).any() or np.isinf(m).any() or np.isinf(q).any():
                # print('inf or nan encountered in posterior, will change to another step size to continue grid search')
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(m, np.diag(1 / q))
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.array([2 * observe_r - 1])
            X = np.array([feature[pull]])
            w = self.optimize(theta, m, q, X, y, explore, max_ite)
            m[:] = w[:]
            p = self.data.logistic(feature[pull].dot(w))
            q += p * (1 - p) * feature[pull] ** 2
            mu[ind] = (mu[ind] * (time[ind] - 1) + observe_r) / time[ind]
        return regret

    def laplacets_op(self, explore_rates, lamda=1, max_ite=1000):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        m = np.zeros(d)
        q = np.ones(d) * lamda

        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)
        explore = explore_rates[index]

        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            if np.isnan(m).any() or np.isnan(q).any() or np.isinf(m).any() or np.isinf(q).any():
                # print('inf or nan encountered in posterior, will change to another step size to continue grid search')
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(m, np.diag(1 / q))
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.array([2 * observe_r - 1])
            X = np.array([feature[pull]])
            w = self.optimize(theta, m, q, X, y, explore, max_ite)
            m[:] = w[:]
            p = self.data.logistic(feature[pull].dot(w))
            q += p * (1 - p) * feature[pull] ** 2
            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret

    def laplacets_tl(self, explore_rates, lamda=1, max_ite=1000):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        m = np.zeros(d)
        q = np.ones(d) * lamda

        explore_lamda = explore_rates
        Kexp = len(explore_lamda)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / ((np.exp(1) - 1) * T)))
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore = explore_lamda[index]

        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0] * K
            if np.isnan(m).any() or np.isnan(q).any() or np.isinf(m).any() or np.isinf(q).any():
                # print('inf or nan encountered in posterior, will change to another step size to continue grid search')
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(m, np.diag(1 / q))
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ts_idx)
            observe_r = self.data.random_sample(t, pull)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.array([2 * observe_r - 1])
            X = np.array([feature[pull]])
            w = self.optimize(theta, m, q, X, y, explore, max_ite)
            m[:] = w[:]
            p = self.data.logistic(feature[pull].dot(w))
            q += p * (1 - p) * feature[pull] ** 2
            logw, p, index = tl_auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_lamda[index]
        return regret