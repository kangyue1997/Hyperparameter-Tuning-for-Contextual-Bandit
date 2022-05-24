from sklearn.linear_model import LogisticRegression
from algo.zooming_ts_tuning import *
import itertools as it

class GLOC:
    def __init__(self, class_context):
        self.data = class_context
        self.T = self.data.T
        self.d = self.data.d
    def grad(self, theta, X, y):
        d = self.d
        g = -y + self.data.logistic(X.dot(theta))
        return g
    def argm(self, theta_prime, A, S, eta):
        n = np.linalg.norm(theta_prime)
        if n <= S:
            return theta_prime
        theta = np.zeros(self.d)
        for ite in range(1000):
            grad = 2*A.dot(theta - theta_prime)
            if np.linalg.norm(grad) <= 10**(-4):
                break
            if ite%100 == 0:
                eta /= 2
            theta -= eta * grad
            n = np.linalg.norm(theta)
            if n>S:
                theta /= (n/S)
        return theta

    def gloc_theoretical_explore(self):
        return np.array([-1]*self.T)

    def gloc_auto(self, inte = [[0,1],[0,1]], eta = 1, S = 1, lamda = 1, eps = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        A = eps * np.identity(d)
        A_inv = 1/eps * np.identity(d)
        V_inv = 1/lamda * np.identity(d)
        theta_hat = np.zeros(d)
        theta = np.zeros(d)
        xz = np.zeros(d)

        c = 13 / 2
        low = np.array([0.7 * a[0] + 0.3 * a[1] for a in inte])
        up = np.array([0.3 * a[0] + 0.7 * a[1] for a in inte])
        cen = [(up - low) * np.random.random_sample(2) + low]
        time = [1]
        rad = [math.sqrt(c * math.log(T))]
        sd = [math.sqrt(2 * c)]
        explore = cen[0]

        beta = trans(2, explore[0], 3)
        k = trans(1, explore[1], 1)

        feature = self.data.fv[0]
        K = len(feature)
        ucb_idx = [0] * K
        for arm in range(K):
            ucb_idx[arm] = feature[arm].dot(theta_hat) + beta * math.sqrt(
                feature[arm].dot(V_inv).dot(feature[arm]))
        pull = np.argmax(ucb_idx)
        observe_r = self.data.random_sample(0, pull)
        gs = self.grad(theta, feature[pull], observe_r)
        regret[0] = self.data.optimal[0] - self.data.reward[0][pull]
        tmp = A_inv.dot(feature[pull])
        A_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
        A += np.outer(feature[pull], feature[pull])
        tmp = V_inv.dot(feature[pull])
        V_inv -= np.outer(tmp, tmp) / (1 + feature[pull].dot(tmp))
        theta_prime = theta - gs / k * A_inv.dot(feature[pull])
        theta = self.argm(theta_prime, A, S, eta)
        xz += feature[pull].dot(theta) * feature[pull]
        theta_hat = V_inv.dot(xz)
        mu = [observe_r]

        for t in range(1,T):
            feature = self.data.fv[t]
            K = len(feature)
            ind, cen, time, mu, rad, sd = auto_tuning(cen, time, rad, sd, c, T, mu, inte)
            explore = cen[ind]
            beta = trans(2, explore[0], 3)
            k = trans(1, explore[1], 1)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + beta * math.sqrt(feature[arm].dot(V_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            gs = self.grad(theta, feature[pull], observe_r)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            tmp = A_inv.dot(feature[pull])
            A_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            A += np.outer(feature[pull], feature[pull])
            tmp = V_inv.dot(feature[pull])
            V_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            theta_prime = theta - gs / k * A_inv.dot(feature[pull])
            theta = self.argm(theta_prime, A, S, eta)
            xz += feature[pull].dot(theta) * feature[pull]
            theta_hat = V_inv.dot(xz)
            mu[ind] = (mu[ind] * (time[ind] - 1) + observe_r) / time[ind]
        return regret

    def gloc_op(self, explore_rates, k = 0.1, eta = 1, S = 1, lamda = 1, eps = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        A = eps * np.identity(d)
        A_inv = 1/eps * np.identity(d)
        V_inv = 1/lamda * np.identity(d)
        theta_hat = np.zeros(d)
        theta = np.zeros(d)
        xz = np.zeros(d)

        Kexp = len(explore_rates)
        s = np.ones(Kexp)
        f = np.ones(Kexp)
        index = np.random.choice(Kexp)
        explore = explore_rates[index]

        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt(feature[arm].dot(V_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            gs = self.grad(theta, feature[pull], observe_r)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            tmp = A_inv.dot(feature[pull])
            A_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            A += np.outer(feature[pull], feature[pull])
            tmp = V_inv.dot(feature[pull])
            V_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            theta_prime = theta - gs / k * A_inv.dot(feature[pull])
            theta = self.argm(theta_prime, A, S, eta)
            xz += feature[pull].dot(theta) * feature[pull]
            theta_hat = V_inv.dot(xz)

            s, f, index = op_tuning(s, f, observe_r, index)
            explore = explore_rates[index]
        return regret


    def gloc_tl(self, explore_rates, k, eta = 1, S = 1, lamda = 1, eps = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        A = eps * np.identity(d)
        A_inv = 1/eps * np.identity(d)
        V_inv = 1/lamda * np.identity(d)
        theta_hat = np.zeros(d)
        theta = np.zeros(d)
        xz = np.zeros(d)

        a = [explore_rates, k]
        explore_rates = list(it.product(*(a[kv] for kv in range(len(a)))))
        Kexp = len(explore_rates)
        logw = np.zeros(Kexp)
        p = np.ones(Kexp) / Kexp
        gamma = min(1, math.sqrt(Kexp * math.log(Kexp) / ((np.exp(1) - 1) * T)))
        # random initial hyper-para
        index = np.random.choice(Kexp)
        explore = explore_rates[index][0]
        k = explore_rates[index][1]

        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + explore * math.sqrt(feature[arm].dot(V_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.data.random_sample(t, pull)
            gs = self.grad(theta, feature[pull], observe_r)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            tmp = A_inv.dot(feature[pull])
            A_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            A += np.outer(feature[pull], feature[pull])
            tmp = V_inv.dot(feature[pull])
            V_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            theta_prime = theta - gs / k * A_inv.dot(feature[pull])
            theta = self.argm(theta_prime, A, S, eta)
            xz += feature[pull].dot(theta) * feature[pull]
            theta_hat = V_inv.dot(xz)

            logw, p, index = tl_auto_tuning(logw, p, observe_r, index, gamma)
            explore = explore_rates[index][0]
            k = explore_rates[index][1]
        return regret
