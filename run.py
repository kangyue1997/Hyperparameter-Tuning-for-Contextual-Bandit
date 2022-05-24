import numpy as np

from algo.LaplaceTS import *
from algo.LinUCB import *
from algo.LinTS import *
from algo.UCB_GLM import *
from algo.SGD_TS import *
from algo.GLM_TSL import *
from algo.GLOC import *
from algo.data_generator import *

import os

import pandas as pd

import concurrent.futures
import warnings

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='auto_tuning_algorithm')
parser.add_argument('-rep', '--rep', type=int, default=10, help='repeat times')
parser.add_argument('-algo', '--algo', type=str, default='linucb', help='can also be lints, glmucb, laplacets, sgdts')
# parser.add_argument('-model', '--model', type=str, default = 'linear', help = 'linear or logistic')
parser.add_argument('-gentype', '--gentype', type=str, default='uniform', help='uniform or normal or movielens')
parser.add_argument('-k', '--k', type=int, default=120, help='number of arms')
parser.add_argument('-context', '--context', type=str, default='True', help='contextual or not')
parser.add_argument('-max_rate', '--max_rate', type=float, default=-1, help='max explore rate')
parser.add_argument('-t', '--t', type=int, default=14000, help='total time')
parser.add_argument('-thetan', '--thetan', type=int, default=1, help='norm of true theta')
parser.add_argument('-d', '--d', type=int, default=25, help='dimension')
parser.add_argument('-c', '--c', type=float, default=6, help='exploration parameter in SGD-TS')
parser.add_argument('-lamda', '--lamda', type=float, default=1, help='lambda, regularization parameter')
parser.add_argument('-delta', '--delta', type=float, default=0.05, help='error probability')
parser.add_argument('-sigma', '--sigma', type=float, default=0.5, help='sub gaussian parameter')
parser.add_argument('-er', '--er', nargs='+', default=[0.5, 1, 3, 5, 7], help='exploration rates')
parser.add_argument('-save', '--save', type=str, default='False', help='save the data or not')
parser.add_argument('-para', '--para', type=str, default='False', help='parallel computing')
# parser.add_argument('-inte', '--inte', nargs = '+', default = [0,1], help = 'exploration intervals')
args = parser.parse_args()

parallel = True if args.para == 'True' else False
rep = args.rep
algo = args.algo  # {'linucb', 'ucbglm', 'lints', 'laplacets', 'sgdts','glmtsl','gloc'}
if algo == 'linucb' or algo == 'lints':
    model = 'linear'
else:
    model = 'logistic'
inte = [0, 1]
if algo == 'sgdts' or algo == 'gloc':
    inte = [[0, 1]] + [[0, 1]]
# model = args.model
gentype = args.gentype
if gentype == 'movie':
    fv = np.loadtxt("data/movie.csv", skiprows=1, delimiter=',')[:, 1:]
    user = np.loadtxt("data/user.csv", skiprows=1, delimiter=',')[:, 1:]
k = args.k
t = args.t
thetan = args.thetan
d = args.d
if gentype == 'movie':
    d = len(user[0])
c = args.c
# data = movielens
lamda = args.lamda
delta = args.delta
sigma = args.sigma
contextual = True
if args.context == 'False':
    contextual = False
ub = 1 / math.sqrt(d)
lb = -1 / math.sqrt(d)
exp_rate = args.er
exp_rate = [float(j) for j in exp_rate]
eta0 = 0.2
save = args.save
if save != 'False':
    save = True
else:
    save = False

methods = {
    'theory': '_theoretical_explore',
    'auto': '_auto',
    'op': '_op',
}

# if not os.path.exists('results/'):
#     os.mkdir('results/')
# if not os.path.exists('results/' + gentype + '/'):
#     os.mkdir('results/' + gentype + '/')
# if not os.path.exists('results/' + gentype + '/' + model + '/'):
#     os.mkdir('results/' + gentype + '/' + algo + '/')
# path = 'results/' + gentype + '/' + algo + '/'
seed = 1

if parallel:
    def func(n0):
        np.random.seed(n0 + seed)
        if gentype == 'movie':
            ind = np.random.choice(len(user), 400, replace=False)
            theta = np.apply_along_axis(np.mean, 0, user[ind, :])
            theta /= np.linalg.norm(theta)
        else:
            theta = np.random.uniform(lb, ub, (d))
        if gentype != 'movie':
            data = context(k, t, d, sigma, theta, theta_norm=thetan, gen_type=gentype, model=model, contextual=True)
        else:
            data = movie(k, t, d, sigma, true_theta=theta, theta_norm=thetan, model=model, fv=fv)
        data.build_bandit()
        if model not in {'linear', 'logistic'}:
            raise ValueError('Not a valid model')

        reg_theory = np.zeros(t)
        reg_auto = np.zeros(t)
        reg_op = np.zeros(t)
        reg_tl = np.zeros(t)

        if algo == 'linucb':
            algo_class = LinUCB(data)
        elif algo == 'lints':
            algo_class = LinTS(data)
        elif algo == 'ucbglm':
            algo_class = UCB_GLM(data)
        elif algo == 'laplacets':
            algo_class = Laplace_TS(data)
        elif algo == 'sgdts':
            algo_class = SGD_TS(data)
        elif algo == 'glmtsl':
            algo_class = GLM_TSL(data)
        elif algo == 'gloc':
            algo_class = GLOC(data)
        fcts = {
            k: getattr(algo_class, algo + methods[k])
            for k, v in methods.items()
        }
        if algo == 'linucb' or algo == 'lints':
            reg_theory += fcts['theory'](lamda=lamda, delta=delta)
            reg_op += fcts['op'](explore_rates=exp_rate, lamda=lamda)
            reg_auto += fcts['auto'](exp_time=30, inte=inte, lamda=lamda)
            reg_tl += fcts['tl'](explore_rates=exp_rate, lamda=lamda)
        elif algo == 'ucbglm':
            reg_theory += fcts['theory'](lamda=lamda, delta=delta)
            reg_op += fcts['op'](explore_rates=exp_rate, lamda=lamda)
            reg_auto += fcts['auto'](exp_time=90, inte=inte, lamda=lamda)
            reg_tl += fcts['tl'](explore_rates=exp_rate, lamda=lamda)
        elif algo == 'laplacets':
            reg_theory += fcts['theory'](lamda=lamda)
            reg_op += fcts['op'](explore_rates=exp_rate)
            reg_auto += fcts['auto'](exp_time=80, inte=inte)
            reg_tl += fcts['tl'](explore_rates=exp_rate)
        elif algo == 'glmtsl':
            reg_theory += fcts['theory'](tau=150, lamda=lamda)
            reg_op += fcts['op'](explore_rates=exp_rate, tau=150, lamda=lamda)
            reg_auto += fcts['auto'](tau=150, inte=inte, lamda=lamda)
            reg_tl += fcts['tl'](explore_rates=exp_rate, tau=150, lamda=lamda)
        elif algo == 'gloc':
            reg_theory += fcts['theory']()
            reg_op += fcts['op'](explore_rates=exp_rate)
            reg_auto += fcts['auto'](inte=inte)
            reg_tl += fcts['tl'](explore_rates=exp_rate, k = exp_rate)
        else:
            reg_theory += fcts['theory'](C=c, eta0=eta0)
            reg_op += fcts['op'](C=c, explore_rates=exp_rate)
            reg_auto += fcts['auto'](C=c, inte=inte)
            reg_tl += fcts['tl'](explore_rates={'eta': exp_rate, 'alpha': exp_rate}, C=c)
        print("theory {0}, auto {1}, op {2}".format(reg_theory[-1], reg_auto[-1], reg_op[-1]))
        return reg_theory, reg_auto, reg_op, reg_tl


    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [nn for nn in range(rep)]
        results = executor.map(func, secs)
        results = list(results)
    res1 = [i[0] for i in list(results)]
    res2 = [i[1] for i in list(results)]
    res3 = [i[2] for i in list(results)]
    res4 = [i[2] for i in list(results)]


    print(algo)
    print('mean')
    print('{0}: reg_theory: {1}'.format(algo, (sum(res1) / rep)[-5:]))
    print('{0}: reg_auto: {1}'.format(algo, (sum(res2) / rep)[-5:]))
    print('{0}: reg_op: {1}'.format(algo, (sum(res3) / rep)[-5:]))
    print('{0}: reg_tl: {1}'.format(algo, (sum(res4) / rep)[-5:]))

    std_theory = [np.std([a[i] for a in res1]) for i in range(t)]
    std_auto = [np.std([a[i] for a in res2]) for i in range(t)]
    std_op = [np.std([a[i] for a in res3]) for i in range(t)]
    std_tl = [np.std([a[i] for a in res4]) for i in range(t)]

    print('\n')
    print('std')
    print('{0}: std_theory: {1}'.format(algo, list(std_theory[-5:])))
    print('{0}: std_auto: {1}'.format(algo, list(std_auto)[-5:]))
    print('{0}: std_op: {1}'.format(algo, list(std_op)[-5:]))
    print('{0}: std_tl: {1}'.format(algo, list(std_tl)[-5:]))
    print('\n')
    print('\n')
    print('\n')

    if save:
        if not os.path.exists('results/'):
            os.mkdir('results/')
        if not os.path.exists('results/' + gentype + '/'):
            os.mkdir('results/' + gentype + '/')
        if not os.path.exists('results/' + gentype + '/' + algo + '/'):
            os.mkdir('results/' + gentype + '/' + algo + '/')
        path = 'results/' + gentype + '/' + algo + '/'
        file = path + str(t) + '_' + str(k) + '_' + str(d) + '_'
        for v in exp_rate:
            file += str(v) + '_'
        file = file[:-1] + '.csv'

        df = pd.DataFrame(
            {'theory': sum(res1) / rep, 'auto': sum(res2) / rep, 'op': sum(res3) / rep, 'tl': sum(res4)/rep ,'std_theory': std_theory,
             'std_auto': std_auto, 'std_op': std_op, 'std_tl':std_tl})
        df.to_csv(file)
else:
    reg_theory = np.zeros(t)
    reg_auto = np.zeros(t)
    reg_op = np.zeros(t)
    reg_tl = np.zeros(t)
    for n0 in range(rep):
        np.random.seed(n0 + seed)
        if gentype == 'movie':
            ind = np.random.choice(len(user), 400, replace=False)
            theta = np.apply_along_axis(np.mean, 0, user[ind, :])
            theta /= np.linalg.norm(theta)
        else:
            theta = np.random.uniform(lb, ub, (d))
        if gentype != 'movie':
            data = context(k, t, d, sigma, theta, theta_norm=thetan, gen_type=gentype, model=model, contextual=True)
        else:
            data = movie(k, t, d, sigma, true_theta=theta, theta_norm=thetan, model=model, fv=fv)
        data.build_bandit()
        if model not in {'linear', 'logistic'}:
            raise ValueError('Not a valid model')


        if algo == 'linucb':
            algo_class = LinUCB(data)
        elif algo == 'lints':
            algo_class = LinTS(data)
        elif algo == 'ucbglm':
            algo_class = UCB_GLM(data)
        elif algo == 'laplacets':
            algo_class = Laplace_TS(data)
        elif algo == 'sgdts':
            algo_class = SGD_TS(data)
        elif algo == 'glmtsl':
            algo_class = GLM_TSL(data)
        elif algo == 'gloc':
            algo_class = GLOC(data)
        fcts = {
            k: getattr(algo_class, algo + methods[k])
            for k, v in methods.items()
        }
        if algo == 'linucb' or algo == 'lints':
            reg_theory += fcts['theory'](lamda=lamda, delta=delta)
            reg_op += fcts['op'](explore_rates=exp_rate, lamda=lamda)
            reg_auto += fcts['auto'](exp_time=30, inte=inte, lamda=lamda)
            reg_tl += fcts['tl'](explore_rates=exp_rate, lamda=lamda)
        elif algo == 'ucbglm':
            reg_theory += fcts['theory'](lamda=lamda, delta=delta)
            reg_op += fcts['op'](explore_rates=exp_rate, lamda=lamda)
            reg_auto += fcts['auto'](exp_time=90, inte=inte, lamda=lamda)
            reg_tl += fcts['tl'](explore_rates=exp_rate, lamda=lamda)
        elif algo == 'laplacets':
            reg_theory += fcts['theory'](lamda=lamda)
            reg_op += fcts['op'](explore_rates=exp_rate)
            reg_auto += fcts['auto'](exp_time=80, inte=inte)
            reg_tl += fcts['tl'](explore_rates=exp_rate)
        elif algo == 'glmtsl':
            reg_theory += fcts['theory'](tau=150, lamda=lamda)
            reg_op += fcts['op'](explore_rates=exp_rate, tau=150, lamda=lamda)
            reg_auto += fcts['auto'](tau=150, inte=inte, lamda=lamda)
            reg_tl += fcts['tl'](explore_rates=exp_rate, tau=150, lamda=lamda)
        elif algo == 'gloc':
            reg_theory += fcts['theory']()
            reg_op += fcts['op'](explore_rates=exp_rate)
            reg_auto += fcts['auto'](inte=inte)
            reg_tl += fcts['tl'](explore_rates=exp_rate, k=exp_rate)
        else:
            reg_theory += fcts['theory'](C=c, eta0=eta0)
            reg_op += fcts['op'](C=c, explore_rates=exp_rate)
            reg_auto += fcts['auto'](C=c, inte=inte)
            reg_tl += fcts['tl'](explore_rates={'eta': exp_rate, 'alpha': exp_rate}, C=c)
        print("theory {0}, auto {1}, op {2}".format(reg_theory[-1], reg_auto[-1], reg_op[-1]))


    if save:
        if not os.path.exists('results/'):
            os.mkdir('results/')
        if not os.path.exists('results/' + gentype + '/'):
            os.mkdir('results/' + gentype + '/')
        if not os.path.exists('results/' + gentype + '/' + algo + '/'):
            os.mkdir('results/' + gentype + '/' + algo + '/')
        path = 'results/' + gentype + '/' + algo + '/'
        file = path + str(t) + '_' + str(k) + '_' + str(d) + '_'
        for v in exp_rate:
            file += str(v) + '_'
        file = file[:-1] + '.csv'

    df = pd.DataFrame(
        {'theory': sum(reg_theory) / rep, 'auto': sum(reg_auto) / rep, 'op': sum(reg_op) / rep, 'tl': sum(reg_tl) / rep,
        })

    df.to_csv(file)