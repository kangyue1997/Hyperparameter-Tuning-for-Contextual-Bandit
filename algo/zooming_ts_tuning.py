import numpy as np
import math


def trans(a,x,k = 2):
    return a * x ** k

def check_1d(cen, rad, inte = [0,1]):
    intervals = [[cen[i]-rad[i], cen[i]+rad[i]] for i in range(len(cen))]
    val = max([i[-1] for i in intervals])
    intervals.sort(key = lambda i : i[0])
    if intervals[0][0] > inte[0]:
        return False, (intervals[0][0]+inte[0])/2
    if val < inte[1]:
        return False, (val+inte[1])/2
    output = [intervals[0]]
    for start, end in intervals[1:]:
        lastEnd = output[-1][1]
        if start <= lastEnd:
            output[-1][1] = max(lastEnd,end)
        else:
            return False, (lastEnd+start)/2
    return True, None

def check_2d(cen,rad,inte = [[0,1],[0,1]]):
    def take_center(x1,r1,x2,r2):
        direct = np.array(x2-x1)
        direct /= np.linalg.norm(direct)
        y1 = x1 + r1*direct
        y2 = x2 - r2*direct
        return (y1+y2)/2
    inte = np.array(inte)
    x0 = inte[0][0]
    x1 = inte[0][1]
    xmid = (x0+x1)/2
    y0 = inte[1][0]
    y1 = inte[1][1]
    ymid = (y0+y1)/2
    vert = np.array([[x0,y0],[xmid,y0],[x1,y0],[x0,ymid],[xmid,ymid],[x1,ymid],[x0,y1],[xmid,y1],[x1,y1]])
    for vertex in vert:
        if all(np.linalg.norm(vertex-cen[i]) > rad[i] for i in range(len(cen))):
            k = np.random.choice(len(cen))
            return False, take_center(vertex,0,cen[k],rad[k])
    return True, None

def auto_tuning(cen,time,rad,sd,c,T,mu,inte):
    d = len(np.array(inte).ravel())/2
    K = len(cen)
    if d > 1:
        res = check_2d(cen,rad,inte)
    else:
        res = check_1d(cen, rad, inte)
    if res[0]:
        tilde_mu = np.random.multivariate_normal(mu, np.diag(sd))
        ind = np.argmax(tilde_mu)
        time[ind] += 1
        rad[ind] *= math.sqrt((time[ind]-1)/time[ind])
        sd[ind] *= math.sqrt((time[ind]-1)/time[ind])
        return ind, cen, time, mu, rad, sd
    else:
        cen = np.concatenate((cen, [res[1]]), axis=0)
        time = np.concatenate((time, [1]), axis=0)
        rad = np.concatenate((rad, [math.sqrt(c*math.log(T))]), axis=0)
        sd = np.concatenate((sd, [math.sqrt(2*c)]), axis=0)
        mu = np.concatenate((mu, [0]), axis=0)
        return K, cen, time, mu, rad, sd


def op_tuning(s, f, reward, index):
    Kexp = len(s)
    r = np.random.binomial(1, max(0, min(reward,1)))
    s[index] += r
    f[index] += (1-r)
    beta = np.array([np.random.beta(s[i], f[i]) for i in range(Kexp)])
    index = np.argmax(beta)
    return s, f, index

def tl_auto_tuning(logw, p, reward, index, gamma):
    Kexp = len(logw)
    # update exp3 components
    logw[index] += (gamma/ Kexp * reward / p[index])
    # run exp3 to determine next hyper-para
    max_logw = np.max(logw)
    w = np.exp(logw - max_logw)
    p = gamma/ Kexp + (1-gamma) * w/sum(w)
    nxt_index = np.random.choice(Kexp, p=p)
    return logw, p, nxt_index
