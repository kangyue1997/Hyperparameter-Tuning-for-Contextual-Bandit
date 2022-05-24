from sklearn.linear_model import LogisticRegression
from algo.zooming_ts_tuning import *


class UCB_GLM:
    def __init__(self, class_context):
        self.data = class_context
        self.T = self.data.T
        self.d = self.data.d

    def ucbglm_theoretical_explore(self, lamda=1, delta=0.1, kappa=0.196 ,explore=-1, exp_time = 30):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(exp_time):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
            tmp = B_inv.dot(feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
        if y[0] == y[1]:
            y[1] = 1 - y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]

        explore_flag = explore
        if explore_flag == -1:
            explore = self.data.sigma / kappa * math.sqrt(
                d / 2 * math.log(1 + 2 * T / d) + math.log(1 / delta))

        for t in range(exp_time, T):
            # when explore = -1, which is impossible, use theoretical value
            # otherwise, it means I have specify a fixed value of explore in the code
            # specify a fixed value for explore is only for grid serach

            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            clf = LogisticRegression(penalty='l2', C=lamda, fit_intercept=False, solver='lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret

    def ucbglm_auto(self, exp_time, inte = [0,1], lamda=1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * lamda
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(exp_time):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1 - y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        B_inv = np.linalg.inv(B)

        # initialization for exp3 algo
        c = 13 / 8
        low = 0.7 * inte[0] + 0.3 * inte[1]
        up = inte[1] - low
        cen = (up - low) * np.random.random_sample(1) + low
        time = [1]
        rad = [math.sqrt(c * math.log(T))]
        sd = [math.sqrt(2 * c)]
        explore = trans(10,cen[0])

        feature = self.data.fv[exp_time]
        K = len(feature)
        ucb_idx = [0] * K
        clf = LogisticRegression(penalty='l2', C=lamda, fit_intercept=False, solver='lbfgs').fit(X, y)
        theta = clf.coef_[0]
        for arm in range(K):
            ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt(
                feature[arm].T.dot(B_inv).dot(feature[arm]))
        pull = np.argmax(ucb_idx)
        observe_r = self.data.random_sample(exp_time, pull)
        tmp = B_inv.dot(feature[pull])
        B += np.outer(feature[pull], feature[pull])
        B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
        y = np.concatenate((y, [observe_r]), axis=0)
        X = np.concatenate((X, [feature[pull]]), axis=0)
        regret[exp_time] = regret[exp_time - 1] + self.data.optimal[exp_time] - self.data.reward[exp_time][pull]

        mu = [observe_r]

        for t in range((exp_time+1), T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            ind, cen, time, mu, rad, sd = auto_tuning(cen, time, rad, sd, c, T, mu, inte)
            explore = trans(10,cen[ind])
            clf = LogisticRegression(penalty='l2', C=lamda, fit_intercept=False, solver='lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

            # update explore rates by auto_tuning
            mu[ind] = (mu[ind]*(time[ind]-1) + observe_r)/time[ind]
        return regret

    def ucbglm_op(self, explore_rates, lamda=1, exp_time=30):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * lamda
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(exp_time):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1 - y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        B_inv = np.linalg.inv(B)

        # initialization for op_tuning
        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)
        explore = explore_rates[index]
        for t in range(exp_time, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            clf = LogisticRegression(penalty='l2', C=lamda, fit_intercept=False, solver='lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

            # update explore rates by op_tuning
            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret


    def ucbglm_tl(self, explore_rates, lamda=1, exp_time=30):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * lamda
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(exp_time):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1 - y[0]
        # random pull in the first two rounds to make sure y[0] != y[1]
        B_inv = np.linalg.inv(B)

        # initialization for EXP3 tuning
        explore_lamda = explore_rates
        Kexp = len(explore_lamda)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / ((np.exp(1) - 1) * T)))
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore = explore_lamda[index]

        for t in range(exp_time, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            clf = LogisticRegression(penalty='l2', C=lamda, fit_intercept=False, solver='lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis=0)
            X = np.concatenate((X, [feature[pull]]), axis=0)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

            # update explore rates by EXP3 tuning
            logw, p, index = tl_auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_lamda[index]
        return regret

    #
    # def glmucb_auto_combined(self, explore_rates, lamdas):
    #     T = self.T
    #     d = self.data.d
    #     regret = np.zeros(T)
    #     theta_hat = np.zeros(d)
    #     y = np.array([])
    #     y = y.astype('int')
    #     X = np.empty([0, d])
    #     theta = np.zeros(d)
    #
    #     xxt = np.zeros((d, d))
    #     for t in range(2):
    #         feature = self.data.fv[t]
    #         K = len(feature)
    #         pull = np.random.choice(K)
    #         observe_r = self.data.random_sample(t, pull)
    #         y = np.concatenate((y, [observe_r]), axis=0)
    #         X = np.concatenate((X, [feature[pull]]), axis=0)
    #         regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
    #         xxt += np.outer(feature[pull], feature[pull])
    #     if y[0] == y[1]:
    #         y[1] = 1 - y[0]
    #     # random pull in the first two rounds to make sure y[0] != y[1]
    #
        # initialization for exp3 algo
        explore_lamda = np.array(np.meshgrid(explore_rates, lamdas)).T.reshape(-1, 2)
        Kexp = len(explore_lamda)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / ((np.exp(1) - 1) * T)))
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore, lamda = explore_lamda[index]
    #
    #     B_inv = np.linalg.inv(xxt + lamda * np.identity(d))
    #     for t in range(2, T):
    #         feature = self.data.fv[t]
    #         K = len(feature)
    #         ucb_idx = [0] * K
    #         clf = LogisticRegression(penalty='l2', C=lamda, fit_intercept=False, solver='lbfgs').fit(X, y)
    #         theta = clf.coef_[0]
    #         for arm in range(K):
    #             ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt(
    #                 feature[arm].T.dot(B_inv).dot(feature[arm]))
    #         pull = np.argmax(ucb_idx)
    #         observe_r = self.data.random_sample(t, pull)
    #
    #         # update explore rates by auto_tuning
    #         logw, p, index = auto_tuning(logw, p, observe_r, index, gamma)
    #         explore, lamda = explore_lamda[index]
    #
    #         xxt += np.outer(feature[pull], feature[pull])
    #         B_inv = np.linalg.inv(xxt + lamda * np.identity(d))
    #
    #         y = np.concatenate((y, [observe_r]), axis=0)
    #         X = np.concatenate((X, [feature[pull]]), axis=0)
    #         regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
    #     return regret