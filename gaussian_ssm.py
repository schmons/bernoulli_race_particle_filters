# gaussian_ssm.py

import numpy as np
import numpy.linalg as la
#import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from bernoulli_resampling import bernoulli_resampling2, bernoulli_resampling_kf
from multinomial_sampling import multinomial_sampling

# Arguments

import argparse
parser = argparse.ArgumentParser("This function runs a selection of particle filters on a Gaussian state space model.")
parser.add_argument("--seed", type=int, default=100, help="Set the seed. Default is 100")
parser.add_argument("--particles", type=int, default=100, help="Set the number of particles used. Default is 100")
parser.add_argument("--reps", type=int, default=1000, help="Set the number of repetitions in the Monte Carlo simulation. Default is 100")
args = parser.parse_args()

# Functions

# simulate Gaussian state space model

def gaussian_ssm(a, sx, sy, N):

    X = np.zeros(N)
    Y = np.zeros(N)
    X[0] = np.sqrt(sx)*np.random.randn(1)
    Y[0] = X[0] + np.sqrt(sy)*np.random.randn(1)

    for i in range(1, N):

        X[i] = a*X[i-1] + np.sqrt(sx)*np.random.randn(1)
        Y[i] = X[i] + np.sqrt(sy)*np.random.randn(1)

    return X, Y

def q_opt(x, y, a, sx, sy):
    
    while(True):
        
        X = a*x + np.sqrt(sx)*np.random.standard_normal(1)
        U = np.random.rand(1)

        if(np.log(U) <= -(y - X)**2/(2*sy)):
            
            return(X) 

def q_opt_fast(x, y, a, sx, sy):
    
    X = 1./2*(a*x+y) + np.sqrt(sx/2)*np.random.standard_normal(1)
    
    return X

def bf(x, x_new, y, a=0.8, sx=5, sy=5):

    X = a*x + np.sqrt(sx)*np.random.standard_normal(1)

    return(-(y - X)**2/(2*sy))

def smc_opt(a, sx, sy, N, Y, pool):
    
    # draw from prior
    T = len(Y)
    
    X_out = np.zeros([N, T])
    traj = np.zeros([N, T])

    indices = np.array([N, T])
    X = np.sqrt(sx)*np.random.randn(N)
    
    # Compute weights
    weights = norm.pdf(Y[0],loc=X, scale=np.sqrt(sy))
    
    # normalise weights
    W = weights/sum(weights)
    
    # resample
    I = multinomial_sampling(W)
    
    #indices[:, 0] = I
    X_out[:, 0] = X[I]
    
    # path weights
    weights_path = np.log(np.mean(weights))#
    Z = np.log(np.mean(weights))

    # Set up trajectory output
    traj[:, 0] = X[I]

    # Effective Sample size
    ESS = np.zeros(T)
    ESS[0] = 1/sum(W**2)
    
    for i in tqdm(range(1, T), ascii=True, ncols=tqdm_col):
        
        # Simulate from prior
        X_new = np.zeros(N)
        weights = np.zeros(N)

        for j in range(N):
            
            weights[j] = 1/np.sqrt(2*np.pi*sy)*np.exp(bf(X_out[j, i-1], X_new[j], Y[i], a, sx, sy))    
    
        # resample
        I, C = bernoulli_resampling_kf(N, X_out[:, i-1], Y[i], pool)
        
        weights_path = weights_path + np.log((N-1)/(sum(C)-1)) - np.log(2*np.pi*sy)/2
        Z += np.log(np.mean(weights))
        
        for j in range(N):
            X_out[j, i] = q_opt(X_out[I[j], i-1], Y[i], a, sx, sy)
        
        # Path storage
        traj[:, 0:i] = traj[I, 0:i]
        traj[:, i] = X_out[I, i]

        ESS[i] = 1/sum(W**2)

    #pool.close()
    #pool.join()
           
    return X_out, weights_path, Z, traj, ESS

def smc_opt_fast(a, sx, sy, N, Y, pool):
    
    # draw from prior
    T = len(Y)
    
    X_out = np.zeros([N, T])
    traj = np.zeros([N, T])

    indices = np.array([N, T])
    X = np.sqrt(sx)*np.random.randn(N)
    
    # Compute weights
    weights = norm.pdf(Y[0],loc=X, scale=np.sqrt(sy))
    
    W = weights/sum(weights)

    # resample
    I = multinomial_sampling(W)
    
    #indices[:, 0] = I
    X_out[:, 0] = X[I]
        
    # path weights
    weights_path = np.log(np.mean(weights))#np.log((N-1)/(sum(C)-1)) 
    
    for i in tqdm(range(1, T), ascii=True, ncols=tqdm_col):
        
        # Simulate from prior
        X_new = np.zeros(N)
    
        # resample
        I, C = bernoulli_resampling_kf(N, X_out[:, i-1], Y[i], pool)
        
        weights_path = weights_path + np.log((N-1)/(sum(C)-1)) - np.log(2*np.pi*sy)/2
        
        for j in range(N):
            X_out[j, i] = q_opt_fast(X_out[I[j], i-1], Y[i], a, sx, sy)
    
    return X_out, weights_path

def smc_opt2(a, sx, sy, N, Y):
    
    # draw from prior
    T = len(Y)
    
    X_out = np.zeros([N, T])
    traj = np.zeros([N, T])

    X = np.sqrt(sx)*np.random.randn(N)#np.zeros(N)
    weights = np.zeros(N)
    
    for j in range(N):
        
        X[j] = q_opt_fast(X_out[j, 0], Y[0], a, sx, sy)
        weights[j] = 1/np.sqrt(2*np.pi*sy)*np.exp(bf(X_out[j, 0], X[j], Y[0], a, sx, sy))
    
    # resample
    W = weights/sum(weights)    
    I = multinomial_sampling(W)

    X_out[:, 0] = X[I]
    
    # path weights
    L = np.log(np.mean(weights))

    # Set up trajectory output
    traj[:, 0] = X[I]

    # Effective Sample size
    ESS = np.zeros(T)
    ESS[0] = 1/sum(W**2)

    for i in tqdm(range(1, T), ascii=True, ncols=tqdm_col):
        
        # Simulate from prior
        X_new = np.zeros(N)
        weights = np.zeros(N)
        
        for j in range(N):
        
            X_new[j]   = q_opt_fast(X_out[j, i-1], Y[i], a, sx, sy)   
            weights[j] = 1/np.sqrt(2*np.pi*sy)*np.exp(bf(X_out[j, i-1], X_new[j], Y[i], a, sx, sy))
    
        # resample
        W = weights/sum(weights)
        
        I = multinomial_sampling(W)
        
        L += np.log(np.mean(weights))

        for j in range(N):
            X_out[j, i] = q_opt(X_out[I[j], i-1], Y[i], a, sx, sy)

        #X_out[:, i] = X_new[I]

        # Path storage
        traj[:, 0:i] = traj[I, 0:i]
        traj[:, i] = X_out[I, i]

        ESS[i] = 1/sum(W**2)
           
    return X_out, L, traj, ESS

def smc_opt2_rej(a, sx, sy, N, Y):
    
    # draw from prior
    T = len(Y)
    
    X_out = np.zeros([N, T])
    traj = np.zeros([N, T])

    X = np.sqrt(sx)*np.random.randn(N)#np.zeros(N)
   
    # Compute weights
    weights = norm.pdf(Y[0],loc=X, scale=np.sqrt(sy))
    
        # resample
    W = weights/sum(weights)    
    I = multinomial_sampling(W)

    X_out[:, 0] = X[I]
    
    # path weights
    L = np.log(np.mean(weights))

    # Set up trajectory output
    traj[:, 0] = X[I]

    # Effective Sample size
    ESS = np.zeros(T)
    ESS[0] = 1/sum(W**2)

    for i in tqdm(range(1, T), ascii=True, ncols=tqdm_col):
        
        # Simulate from prior
        X_new = np.zeros(N)
        weights = np.zeros(N)
        
        for j in range(N):
        
            #X_new[j]   = q_opt(X_out[j, i-1], Y[i], a, sx, sy)   
            weights[j] = 1/np.sqrt(2*np.pi*sy)*np.exp(bf(X_out[j, i-1], X_new[j], Y[i], a, sx, sy))
    
        # resample
        W = weights/sum(weights)
        
        I = multinomial_sampling(W)
        
        L += np.log(np.mean(weights))

        for j in range(N):
            X_out[j, i] = q_opt(X_out[I[j], i-1], Y[i], a, sx, sy)

        #X_out[:, i] = X_new[I]

        # Path storage
        traj[:, 0:i] = traj[I, 0:i]
        traj[:, i] = X_out[I, i]

        ESS[i] = 1/sum(W**2)
           
    return X_out, L, traj, ESS

def smc_opt_fast_rej(a, sx, sy, N, Y, pool):
    
    # draw from prior
    T = len(Y)
    
    X_out = np.zeros([N, T])
    traj = np.zeros([N, T])

    indices = np.array([N, T])
    X = np.sqrt(sx)*np.random.randn(N)
    
    # Compute weights
    weights = norm.pdf(Y[0],loc=X, scale=np.sqrt(sy))
    
    W = weights/sum(weights)

    # resample
    I = multinomial_sampling(W)
    
    #indices[:, 0] = I
    X_out[:, 0] = X[I]
        
    # path weights
    weights_path = np.log(np.mean(weights))#
    
    for i in tqdm(range(1, T), ascii=True, ncols=tqdm_col):
        
        # Simulate from prior
        X_new = np.zeros(N)
    
        # resample
        I, C = bernoulli_resampling_kf(N, X_out[:, i-1], Y[i], pool)
        
        weights_path = weights_path + np.log((N-1)/(sum(C)-1)) - np.log(2*np.pi*sy)/2
        
        for j in range(N):
            X_out[j, i] = q_opt(X_out[I[j], i-1], Y[i], a, sx, sy)
    
    return X_out, weights_path



def main(reps, particles):

    T = 50
    #N = 50
    sx = 5
    sy = 5
    a  = .8
    # simulate model
    np.random.seed(args.seed)
    X, Y = gaussian_ssm(a, sx, sy, T)

    print("==================================================")
    print("Optimal Proposal with Bernoulli Resampling")

    L_opt = np.zeros(reps)
    Z_opt = np.zeros(reps)

    np.random.seed(args.seed)
    for i in tqdm(range(reps), ascii=True, desc="BRPF", ncols=tqdm_col):
        S, L_opt[i], Z_opt[i], traj, _ = smc_opt(a, sx, sy, particles, Y,pool=None)

    print("\r\nBernoulli resampling - normalising constant")
    print(np.mean(L_opt))
    print(np.std(L_opt))

    print("\r\n==================================================")
    print("Optimal Proposal with Multinomial Resampling")

    L_opt2 = np.zeros(reps)

    np.random.seed(args.seed)
    for i in tqdm(range(reps), ascii=True, desc="RWPF", ncols=tqdm_col):
        S, L_opt2[i], traj, _ = smc_opt2_rej(a, sx, sy, particles, Y)

    print("\r\nMultinomial resampling - normalising constant")
    print(np.mean(L_opt2))
    print(np.std(L_opt2))

    # investigating complexity
    print("End of computations")

    return 0

def complexity():

    T = 50
    #N = 50
    sx = 5
    sy = 5
    a  = .8
    # simulate model
    np.random.seed(args.seed)
    X, Y = gaussian_ssm(a, sx, sy, T)

    NN = np.arange(500, 10000, 500)

    times_mult = np.zeros(len(NN))
    times_ber = np.zeros(len(NN))
    times_ber_par = np.zeros(len(NN))

    num_processes = 16
    print("Number of processes:")
    print(num_processes)
    pool = Pool(processes=num_processes)

    for i in tqdm(range(len(NN)), ascii=True, ncols=tqdm_col):

        start_time = time.time()
        smc_opt2(a, sx, sy, NN[i], Y)
        times_mult[i] = time.time() - start_time

        start_time = time.time()
        smc_opt_fast(a, sx, sy, NN[i], Y, pool=None)
        times_ber[i] = time.time() - start_time

        start_time = time.time()
        smc_opt_fast(a, sx, sy, NN[i], Y, pool)
        times_ber_par[i] = time.time() - start_time

    sns.set_context("talk", rc={"lines.linewidth": 3})
    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(NN, times_mult, label="Multinomial")
    plt.plot(NN, times_ber, label="Bernoulli race", linestyle="-.")
    plt.plot(NN, times_ber_par, label="Bernoulli race/16 processes", linestyle="--")
    plt.ylabel("Runtime in seconds")
    plt.xlabel("Number of particles")
    plt.autoscale()
    plt.legend()
    plt.savefig("kf_complexity.png")

    NN = np.arange(500, 10000, 500)

    times_mult = np.zeros(len(NN))
    times_ber = np.zeros(len(NN))
    times_ber_par = np.zeros(len(NN))

    num_processes = 16
    print("Number of processes:")
    print(num_processes)
    pool = Pool(processes=num_processes)

    for i in tqdm(range(len(NN)), ascii=True, ncols=tqdm_col):

        start_time = time.time()
        smc_opt2_rej(a, sx, sy, NN[i], Y)
        times_mult[i] = time.time() - start_time

        start_time = time.time()
        smc_opt_fast_rej(a, sx, sy, NN[i], Y, pool=None)
        times_ber[i] = time.time() - start_time

        start_time = time.time()
        smc_opt_fast_rej(a, sx, sy, NN[i], Y, pool)
        times_ber_par[i] = time.time() - start_time

    sns.set_context("talk", rc={"lines.linewidth": 3})
    plt.figure(num=None, figsize=(16, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(NN, times_mult, label="Multinomial")
    plt.plot(NN, times_ber, label="Bernoulli race", linestyle="-.")
    plt.plot(NN, times_ber_par, label="Bernoulli race/16 processes", linestyle="--")
    plt.ylabel("Runtime in seconds")
    plt.xlabel("Number of particles")
    plt.autoscale()
    plt.legend()
    plt.savefig("kf_complexity:rej.png")


    return 0

def trajectory_test(particles, reps):
    
    T = 50
    #N = 50
    sx = 5
    sy = 5
    a  = .8
    start_time = time.time()

    np.random.seed(args.seed)
    X, Y = gaussian_ssm(a, sx, sy, T)

    np.random.seed(args.seed)

    f1 = np.zeros(reps)
    g1 = np.zeros(reps)
    h1 = np.zeros(reps)
    k1 = np.zeros(reps)
    vs = np.zeros(reps)
    
    for i in tqdm(range(reps), ascii=True, ncols=tqdm_col):
        S, L, Z, traj, _ = smc_opt(a, sx, sy, particles, Y, pool=None)
        f1[i] = np.mean(np.mean(traj, axis=1))
        g1[i] = np.mean(la.norm(traj, axis=1))
        h1[i] = np.mean(S[:, T-1])
        k1[i] = np.var(S[:, T-1])
        vs[i] = np.std(la.norm(traj, axis=1))


    print("\r\n Variance of mean function")
    print(np.std(f1))
    print(np.std(g1))
    print(np.std(h1))
    print(np.std(k1))
    print(np.mean(vs))

    means = np.zeros(T)
    var   = np.zeros(T)

    for i in range(T):

        means[i] = np.mean(S[:,i])
        var[i]   = np.var(S[:, i])

    plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(means)
    v = np.arange(T)

    plt.scatter(v, Y, label="Observations")
    plt.ylim(-10, 6)
    plt.suptitle("Bernoulli resampling")

    #for i in range(20):
        
    #    plt.plot(v, traj[i, :])
    
    plt.legend()
    plt.savefig("opt_ber.png")

    ##############################################################################

    np.random.seed(args.seed)

    f2 = np.zeros(reps)
    g2 = np.zeros(reps)
    h2 = np.zeros(reps)
    k2 = np.zeros(reps)
    vs = np.zeros(reps)

    print("\r\n==================================================\r\n")

    for i in tqdm(range(reps), ascii=True, ncols=tqdm_col):
        S, L, traj,_ = smc_opt2_rej(a, sx, sy, particles, Y)
        f2[i] = np.mean(np.mean(traj, axis=1))
        g2[i] = np.mean(la.norm(traj, axis=1))
        h2[i] = np.mean(S[:, T-1])
        k2[i] = np.var(S[:, T-1])
        vs[i] = np.std(la.norm(traj, axis=1))
    
    print("\r\n Variance of mean function")
    print(np.std(f2))
    print(np.std(g2))
    print(np.std(h2))
    print(np.std(k2))
    print(np.mean(vs))

    means = np.zeros(T)
    var   = np.zeros(T)

    for i in range(T):

        means[i] = np.mean(S[:,i])
        var[i]   = np.var(S[:, i])

    plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(means)
    v = np.arange(T)

    plt.scatter(v, Y, label="Observations")
    plt.ylim(-10, 6)
    plt.suptitle("Multinomial resampling")

    #for i in range(20):
        
    #    plt.plot(v, traj[i, :])

    plt.legend()
    plt.savefig("opt_mult.png")

    return 0

#def test_function(x):

if __name__ == "__main__":

    tqdm_col = 100
    #complexity()
    trajectory_test(args.particles, args.reps)
    main(args.reps, args.particles)
