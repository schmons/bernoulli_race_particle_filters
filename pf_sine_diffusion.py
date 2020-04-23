# gaussian_ssm.py

import numpy as np
import numpy.linalg as la
import pandas as pd
import time
from multiprocessing import Pool
from tqdm import tqdm
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("poster")

from bernoulli_resampling import bernoulli_resampling2, bernoulli_resampling
from multinomial_sampling import multinomial_sampling
from sdesim import *

# Arguments

import argparse
parser = argparse.ArgumentParser("This function runs a selection of particle filters on a Gaussian state space model.")
parser.add_argument("--seed", type=int, default=100, help="Set the seed. Default is 100")
parser.add_argument("--particles", type=int, default=100, help="Set the number of particles used. Default is 100")
parser.add_argument("--reps", type=int, default=1000, help="Set the number of repetitions in the Monte Carlo simulation. Default is 100")
args = parser.parse_args()

# Helpers

def q_opt(x, y, delta, noise=0.5):
    
    while True:
        
        X = rejection_sampler(x, delta)
        U = np.random.rand(1)
        
        if np.log(U) <= -(y - X)**2/(2*noise**2):
                   
            return X    

def opt_weight(x, y, delta, noise=0.5):
    
    X = rejection_sampler(x, delta)
    
    return -(y - X)**2/(2*noise**2)

# Set up particle filters 

# Standard Random Weight Particle Filter
def sine_smc_rw(N, Y, t_grid, noise=0.5):
    
    # draw from prior
    T = len(Y)
    
    X_out = np.zeros([N, T])
    traj = np.zeros([N, T])
    X = noise*np.random.standard_normal(size=N)
    
    # Track normalising constants
    Z = np.zeros(T)
    
    # Compute weights
    weights = norm.pdf(Y[0],loc=X,scale=noise)
    
    # normalise weights
    W = weights/np.sum(weights)
    
    weights_path = np.zeros(T)
    weights_path[0] = np.log(np.mean(weights))
    
    # resample    
    I = multinomial_sampling(W)
    X_out[:, 0] = X[I]
    #X_out[:, 0] = np.random.choice(X, p=W, size=N)
    
    # Collect ESS 
    ESS = np.zeros(T)
    ESS[0] = 1/sum(W**2)
    
    # Set up trajectory output
    traj[:, 0] = X[I]
    
    for i in tqdm(range(1, T), leave=False, ascii=True, ncols=100):
        
        Delta = t_grid[i]-t_grid[i-1]
        
        # Propose simulate forward
        X_new = X_out[:,i-1] + Delta*np.sin(X_out[:,i-1]) + np.sqrt(Delta)*np.random.standard_normal(N)
        
        # Compute weights
        pe = np.zeros(N)
        
        for j in range(N):
            pe[j] = poisson_estimator2(X_out[j,i-1], X_new[j], t=Delta, la=9/8)
            #pe[j] = exact_coin(X_out[j,i-1], X_new[j],Delta, la=9/8)
        
        ratio = norm.pdf(X_new, loc=X_out[:, i-1], scale=np.sqrt(Delta))/norm.pdf(X_new, loc=(Delta*np.sin(X_out[:,i-1]) + X_out[:, i-1]), scale=np.sqrt(Delta))
        weights = ratio*pe*norm.pdf(Y[i],loc=X_new,scale=noise)*np.exp(-np.cos(X_new)+np.cos(X_out[:, i-1]))*np.exp((9/8-5/8)*Delta)
        #
        #/norm.pdf(X_new, loc=X_out[:, i-1], scale=np.sqrt(t_grid[i] - t_grid[i-1]))
        
        W = weights/np.sum(weights)
        
        weights_path[i] = np.log(np.mean(weights))
        #X_out[:, i] = np.random.choice(X_new, p=W, size=N)
        #print(W)
        
        I = multinomial_sampling(W)
        X_out[:, i] = X_new[I]
         
        # Collect ESS
        ESS[i] = 1/sum(W**2)
        
        # Path storage
        traj[:, 0:i] = traj[I, 0:i]
        traj[:, i] = X_new[I]

    return X_out, sum(weights_path), ESS, traj

# Bernoulli Particle Filter
def sine_smc_bernoulli(N, Y, t_grid, noise=0.5, pool=None):
    
    # draw from prior
    T = len(Y)

    #num_processes = 8
    #pool = Pool(processes=num_processes)
    
    X_out = np.zeros([N, T])
    traj = np.zeros([N, T])
    X = noise*np.random.randn(N)
    

    pe = np.zeros(N)

    # Compute weights
    weights = norm.pdf(Y[0],loc=X,scale=noise)
    
    # normalise weights
    W = weights/np.sum(weights)#*np.random.beta(5, 1, 1)*6/5
    
    # resample    
    I = multinomial_sampling(W)
    #I, C = bernoulli_resampling2(N, W)
    #Z = np.log(1/np.mean(C)) 
    X_out[:, 0] = X[I]
    
    Z = np.zeros(T)
    Z[0] = np.log(np.mean(weights))
    weights_path = np.log(np.sum(weights))
    
    # Collect ESS 
    ESS = np.zeros(T)
    ESS[0] = 1/sum(W**2)
    
    # Set up trajectory output
    traj[:, 0] = X[I]
    
    for i in tqdm(range(1, T), leave=False, ascii=True, ncols=100):
        
        # Propose simulate forward
        Delta = t_grid[i] - t_grid[i-1]
        
        X_new = X_out[:,i-1] + Delta*np.sin(X_out[:,i-1]) + np.sqrt(Delta)*np.random.randn(N)
            
        for j in range(N):
            pe[j] = poisson_estimator2(X_out[j,i-1], X_new[j], t=Delta, la=9/8)

        # Compute weights  
        #poisson_estimator(X_out[j,i-1], X_new[j], t=t_grid[i]-t_grid[i-1], la=9/8)
        ratio = norm.pdf(X_new, loc=X_out[:, i-1], scale=np.sqrt(Delta))/norm.pdf(X_new, loc=Delta*np.sin(X_out[:,i-1]) + X_out[:, i-1], scale=np.sqrt(Delta))
        
        weights = ratio*norm.pdf(Y[i], loc=X_new, scale=noise)*np.exp(-np.cos(X_new)+np.cos(X_out[:,i-1]))*np.exp((9/8 - 5/8)*Delta)
        
        W = weights/np.sum(weights)
        
        I, C = bernoulli_resampling(W, poisson_pgf_coin, X_out[:,i-1], X_new, Delta, pool)

        Z[i] = np.log((N-1)/(sum(C)-1)) + np.log(np.mean(weights)) 

        X_out[:, i] = X_new[I]
        
        # Collect ESS 
            
        #for j in range(N):
        #    pe[j] = poisson_estimator2(X_out[j,i-1], X_new[j], t=Delta, la=9/8)

        #weights = weights*pe

        # Update weights
        #weights_path += np.log(np.mean(weights))

        #W = weights/sum(weights)
        #ESS[i] = 1/sum(W**2)#((N-1)/(sum(C)-1))**2/((N-1)/(sum(C)-1))*(sum(weights))
        
        # Path storage
        traj[:, 0:i] = traj[I, 0:i]
        traj[:, i] = X_new[I]
    
    #pool.close()
    #pool.join()
           
        
    return X_out, sum(Z), ESS, traj

def sine_smc_bootstrap(N, Y, t_grid, noise=0.5):
    
    # draw from prior
    T = len(Y)
    
    X_out = np.zeros([N, T])
    X = noise*np.random.randn(N)
    
    # Compute weights
    weights = norm.pdf(Y[0],loc=X,scale=noise)
    
    # normalise weights
    W = weights/np.sum(weights)
    
    # resample    
    #I, C = bernoulli_resampling2(np.ones(N)/N, weights)
    I = multinomial_sampling(W)
    #Z = np.log(1/np.mean(C)) 
    X_out[:, 0] = X[I]
    
    #L = np.log((N-1)/(sum(C)-1))
    Z = np.log(np.mean(weights))
    
    # Collect ESS 
    ESS = np.zeros(T)
    ESS[0] = 1/sum(W**2)
    
    for i in tqdm(range(1, T), leave=False, ascii=True, ncols=100):
        
        # Propose simulate forward
        Delta = t_grid[i] - t_grid[i-1]
        
        X_new = np.zeros(N)
        
        for j in range(N):
            X_new[j] = rejection_sampler(X_out[j,i-1], Delta)
            #t, S = euler_sin(X_out[j,i-1], Delta/1000, 1000*Delta)
            #S, t = exact_sampler(X_out[j,i-1], Delta)
            #S, t = retrospective_sampler2(X_out[j,i-1], Delta)
            #X_new[j] = S[-1] 
        
        # Compute weights
        weights = norm.pdf(Y[i],loc=X_new,scale=noise)
        
        W    = weights/np.sum(weights)
        
        #I, C = bernoulli_resampling2(np.ones(N)/N, weights)
        I = multinomial_sampling(W)
        #L += np.log((N-1)/(sum(C)-1))
        Z  = Z + np.log(np.mean(weights)) 

        X_out[:, i] = X_new[I]
        # Collect ESS 
        ESS[i] = 1/sum(W**2)
           
    return X_out, Z, ESS



# main script

def main(particles, reps, seed):

    #sns.set_context('poster')

    T = 30
    noise = 5

    np.random.seed(seed)
    #Y, S, t = sine_ssm_long(0, 10)
    Y, S, t = sine_ssm_alt(0, T, noise)

    # plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    # plt.plot(t, S)
    # plt.scatter(t, Y)
    # plt.savefig("sine_dif_ssm.png")

    start_time = time.time()

    np.random.seed(seed)
    print("Number of particles:", particles)
    test, L1, ESS1, _ = sine_smc_rw(particles, Y, t, noise)
    print("--- %s seconds ---" % (time.time() - start_time))

    means1 = np.zeros(len(t))

    for i in range(len(t)):

        means1[i] = np.mean(test[:,i])

    np.random.seed(seed)
    print("Number of particles:", 1000)
    test_high, L, ESS, _ = sine_smc_rw(1000, Y, t, noise)
    print("--- %s seconds ---" % (time.time() - start_time))

    means = np.zeros(len(t))

    for i in range(len(t)):

        means[i] = np.mean(test_high[:,i])

    start_time = time.time()

    np.random.seed(seed)
    print("Number of particles:", particles)
    test2, L2, ESS2, _ = sine_smc_bernoulli(particles, Y, t, noise)
    print("--- %s seconds ---" % (time.time() - start_time))

    means2 = np.zeros(len(t))

    for i in range(len(t)):

        means2[i] = np.mean(test2[:,i])



    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(t, S, linestyle="-.", label="State")
    plt.scatter(t, Y)
    plt.plot(t, means1, label="RWPF100")
    plt.plot(t, means2, label="BerPF100")
    plt.plot(t, means, label="RWPF1000")
    plt.legend()
    plt.savefig("tracking.png")

    start_time = time.time()

    np.random.seed(seed)
    print("Number of particles:", particles)
    test3, Z, ESS3 = sine_smc_bootstrap(particles, Y, t, noise)
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(L)
    print(Z)    

    ## Sde means

    means3 = np.zeros(len(t))

    for i in range(len(t)):

        means3[i] = np.mean(test3[:, i])

    k = 10
    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k') #
    sns.kdeplot(test[:, len(Y)-k], label="RWPF100", shade=True)
    sns.kdeplot(test2[:, len(Y)-k], label="BRPF100", shade=True)
    sns.kdeplot(test_high[:, len(Y)-k], label="RWPF1000", shade=True)
    plt.legend()
    plt.savefig("kdeplot_sde.png")

    import numpy.linalg as la

    T = len(Y)
    np.random.seed(seed)

    print("Compare Test functions")
    print("Multinomial resampling")

    f1 = np.zeros(reps)
    g1 = np.zeros(reps)
    h1 = np.zeros([reps, T])
    k1 = np.zeros([reps, T])

    for i in tqdm(range(reps), ascii=True, ncols=100):
        S, L, Z, traj = sine_smc_rw(particles, Y, t, noise)
        f1[i] = np.mean(np.mean(traj, axis=1))
        g1[i] = np.mean(la.norm(traj, axis=1))
        for j in range(T):
            h1[i, j] = np.mean(S[:, j])
            k1[i, j]= np.var(S[:, j])

    print("\r\n Variance of mean function")
    print(np.std(f1))
    print(np.std(g1))
    h1_means = np.std(h1, axis=0)
    k1_means = np.std(k1, axis=0)
    print(h1_means[-1])
    print(k1_means[-1])


    print("\nBernoulli resampling\n")


    np.random.seed(seed)

    f2 = np.zeros(reps)
    g2 = np.zeros(reps)
    h2 = np.zeros([reps, T])
    k2 = np.zeros([reps, T])

    for i in tqdm(range(reps), ascii=True, ncols=100):
        S, L, ess, traj = sine_smc_bernoulli(particles, Y, t, noise)
        f2[i] = np.mean(np.mean(traj, axis=1))
        g2[i] = np.mean(la.norm(traj, axis=1))
        for j in range(T):
            h2[i, j] = np.mean(S[:, j])
            k2[i, j] = np.var(S[:, j])

    print("\r\n Variance of mean function")
    print(np.std(f2))
    print(np.std(g2))
    h2_means = np.std(h2, axis=0)
    k2_means = np.std(k2, axis=0)
    print(h2_means[-1])
    print(k2_means[-1])

    print("==================================================\n")
    print("Compare normalising constant")

    #reps = 1000

    Z1    = np.zeros(reps)

    np.random.seed(seed)
    for i in tqdm(range(reps), ascii=True, ncols=100):
        B, Z1[i],_ , _ = sine_smc_rw(particles, Y, t, noise)

    print(np.mean(Z1))
    print(np.std(Z1))

    sns.set_style('whitegrid')
    sns.kdeplot(Z1, bw=0.3)

    Z2   = np.zeros(reps)
    L2   = np.zeros(reps)

    np.random.seed(seed)
    for i in tqdm(range(reps), ascii=True, ncols=100):
        B, L2[i], _, _ = sine_smc_bernoulli(particles, Y, t, noise)

    print("Bernoulli Estimate")
    print(np.mean(L2))
    print(np.std(L2))

    sns.set_style('whitegrid')
    sns.kdeplot(L2, bw=0.3)

    Z3    = np.zeros(reps)
    L3    = np.zeros(reps)

    np.random.seed(seed)
    for i in tqdm(range(reps), ascii=True, ncols=100):
        B, Z3[i], _ = sine_smc_bootstrap(particles, Y, t, noise)

    print(np.mean(Z3))
    print(np.std(Z3))

    reps = 10

    Z4    = np.zeros(reps)
    L4    = np.zeros(reps)

    return 0


def complexity(seed):

    #sns.set_context('poster')

    T = 30
    noise = 5

    np.random.seed(seed)
    #Y, S, t = sine_ssm_long(0, 10)
    Y, S, t = sine_ssm_alt(0, T, noise)

    NN = np.arange(500, 10000, 500)

    times_mult = np.zeros(len(NN))
    times_ber = np.zeros(len(NN))
    times_ber_par = np.zeros(len(NN))

    num_processes = 16
    print(num_processes)
    pool = Pool(processes=num_processes)
    

    for i in tqdm(range(len(NN)), ascii=True, ncols=100):

        start_time = time.time()
        sine_smc_rw(NN[i], Y, t, noise)
        times_mult[i] = time.time() - start_time

        start_time = time.time()
        sine_smc_bernoulli(NN[i], Y, t, noise, pool=None)
        times_ber[i] = time.time() - start_time

        start_time = time.time()
        sine_smc_bernoulli(NN[i], Y, t, noise, pool)
        times_ber_par[i] = time.time() - start_time

    sns.set_context("talk", rc={"lines.linewidth": 2})
    plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(NN, times_mult, label="Multinomial")
    plt.plot(NN, times_ber, label="Bernoulli race", linetype="--")
    plt.plot(NN, times_ber_par, label="Bernoulli race/16 processes", linetype="-.")
    plt.ylabel("Runtime in seconds")
    plt.xlabel("Number of particles")
    #plt.xticks(NN)
    plt.autoscale()
    plt.legend()
    plt.savefig("sine_ssm_complexity.png")

    return 0

if __name__=="__main__":

    #complexity(args.seed)
    main(args.particles, args.reps, args.seed)

