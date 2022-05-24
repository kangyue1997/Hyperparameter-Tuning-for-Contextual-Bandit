from algo.zooming_ts_tuning import *


class LinUCB:
    def __init__(self, class_context):
        self.data = class_context
        self.T = self.data.T
        self.d = self.data.d

    def linucb_theoretical_explore(self, lamda=1, delta=0.1, explore=-1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)
        explore_flag = explore
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K

            # when explore = -1, which is impossible, use theoretical value
            # otherwise, it means I have specify a fixed value of explore in the code
            # specify a fixed value for explore is only for grid serach
            if explore_flag == -1:
                # explore = self.data.sigma * math.sqrt(
                #     d * math.log((t * self.data.max_norm ** 2 / lamda + 1) / delta)) + math.sqrt(lamda)
                explore = self.data.sigma * math.sqrt(
                    d * math.log((t / lamda + 1) / delta)) + math.sqrt(lamda)*self.data.theta_norm
            else:
                explore = explore_flag
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret

    def linucb_auto(self, exp_time, inte = [0,1], lamda=1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)


        for t in range(exp_time):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.data.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B += np.outer(feature[pull], feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            xr += feature[pull] * observe_r
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

        # initialization for exp3 algo
        # the possible choices for C is in J
        # the following two lines are an ideal set of "explore_rates"
        # min_rate, max_rate = 0, 2 * (int(math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1)
        # J = np.arange(min_rate, max_rate, explore_interval_length)
        c = 13/8
        low = 0.7*inte[0] + 0.3*inte[1]
        up = inte[1] - low
        cen = (up-low)*np.random.random_sample(1)+low
        time = [1]
        rad = [math.sqrt(c * math.log(T))]
        sd = [math.sqrt(2*c)]
        feature = self.data.fv[exp_time]
        K = len(feature)
        explore = trans(2,cen[0])
        ucb_idx = [0] * K

        for arm in range(K):
            ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt(
                feature[arm].T.dot(B_inv).dot(feature[arm]))
        pull = np.argmax(ucb_idx)
        observe_r = self.data.random_sample(exp_time, pull)

        # update linucb
        tmp = B_inv.dot(feature[pull])
        B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
        xr += feature[pull] * observe_r
        theta_hat = B_inv.dot(xr)
        regret[exp_time] = regret[exp_time - 1] + self.data.optimal[exp_time] - self.data.reward[exp_time][pull]

        mu = [observe_r]

        for t in range((exp_time+1),T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K
            ind, cen, time, mu, rad, sd = auto_tuning(cen, time, rad, sd, c, T, mu, inte)
            explore = trans(2,cen[ind])

            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)

            # update linucb
            tmp = B_inv.dot(feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

            mu[ind] = (mu[ind]*(time[ind]-1) + observe_r)/time[ind]
            # update explore rates by auto_tuning
        return regret

    def linucb_op(self, explore_rates, lamda=1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        B = np.identity(d) * lamda
        B_inv = np.identity(d) / lamda
        theta_hat = np.zeros(d)

        # initialization for exp3 algo
        # the possible choices for C is in J
        # the following two lines are an ideal set of "explore_rates"
        # min_rate, max_rate = 0, 2 * (int(math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1)
        # J = np.arange(min_rate, max_rate, explore_interval_length)
        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)
        explore = explore_rates[index]

        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K

            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)

            # update linucb
            tmp = B_inv.dot(feature[pull])
            B_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]

            # update explore rates by auto_tuning
            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret

    def linucb_tl(self, explore_rates, lamda = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        xr = np.zeros(d)
        theta_hat = np.zeros(d)

        # initialization for exp3 algo
        # the possible choices for C is in J
        # the following two lines are an ideal set of "explore_rates"
        # min_rate, max_rate = 0, 2 * (int(math.sqrt(d * math.log(T**2+T) ) + math.sqrt(lamda)) + 1)
        # J = np.arange(min_rate, max_rate, explore_interval_length)
        explore_lamda = explore_rates
        Kexp = len(explore_lamda)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / ((np.exp(1) - 1) * T)))
        # random initial explore rate
        index = np.random.choice(Kexp)
        explore = explore_lamda[index]

        xxt = np.zeros((d, d))
        B_inv = np.identity(d) / lamda
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0] * K

            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt(
                    feature[arm].T.dot(B_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)

            # update explore rates by auto_tuning
            logw, p, index = tl_auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_lamda[index]

            # update linucb
            xxt += np.outer(feature[pull], feature[pull])
            B_inv = np.linalg.inv(xxt + lamda * np.identity(d))
            xr += feature[pull] * observe_r
            theta_hat = B_inv.dot(xr)
            regret[t] = regret[t - 1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret