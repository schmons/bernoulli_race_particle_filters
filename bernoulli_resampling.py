import numpy as np
from scipy.stats import multivariate_normal
#from joblib import Parallel, delayed
from multiprocessing import Pool

from alias import *

def bernoulli_race(p, J, q, x, y, t):
    
    count = 0
    
    while(True):
        
        I = alias_draw(J, q)
        count += 1        
        pest  = p(x[I], y[I], t, la=9/8)
        #pest  = p(x[I], y, t)
        if np.random.random() <= pest:
            
            return np.array([I, count])


def bernoulli_resampling(c, p, x, y, t, pool):

    N = len(c)

    J, q = alias_setup(c)

    if pool is None:
        out = np.zeros(N)
        count = np.zeros(N)

        for i in range(N):

            out[i], count[i] = bernoulli_race(p, J, q, x, y, t)

        return out.astype(int), count

    else:
        out = pool.starmap(bernoulli_race, [(p, J, q, x, y, t)] * N)
        out = np.array(out)
        return out[:, 0].astype(int), out[:, 1]

def bernoulli_race2(N, p):
    
    count = 0
    np.random.seed()

    while(True):
        
        I = np.random.random_integers(0, N-1, size=1)
        count += 1        

        if np.random.rand(1) < p[I]:
            
            return np.array([I, count])#I, count


def bernoulli_resampling2(N, p, pool=None, parallel=False):
    
    if parallel==False:

        out = np.zeros(N)
        count = np.zeros(N)

        for i in range(N):
            
            out[i], count[i] = bernoulli_race2(N, p)
            
        return out.astype(int), count

    else: 
        
        assert pool is not None, "pool needs to be provided if parallel is used"
        #J, q = alias_setup(c)
        out = pool.starmap(bernoulli_race2, [(N, p)] * N)
        #out = Parallel(n_jobs=8)(delayed(bernoulli_race2)(N, p) for i in range(N))
        out = np.array(out)

    return out[:,0].astype(int), out[:,1] 

# Resampling for Cox process particle filter

def bernoulli_race3(J, q, N, x, t, theta, sigma, p):

    count = 0

    while(True):

        I = alias_draw(J, q)
        
        #I = np.random.random_integers(0, N-1, size=1)
        count += 1        
        pest = p(x[I , :, :], t, theta, sigma)

        if np.random.random() <= pest:
            
            return np.array([I, count])


def bernoulli_resampling3(c, x, t, theta, sigma, p, pool):

    N = len(c)

    J, q = alias_setup(c)

    if pool is None:
        out = np.zeros(N)
        count = np.zeros(N)

        for i in range(N):

            out[i], count[i] = bernoulli_race3(J, q, N, x, t, theta, sigma, p)

        return out.astype(int), count

    else:
        out = pool.starmap(bernoulli_race3, [(J, q, N, x, t, theta, sigma, p)] * N)
        out = np.array(out)

        return out[:,0].astype(int), out[:,1]

# Resampling for Cox process particle filter

def bernoulli_race4(J, q, N, x, y, t0, t1, theta, sigma, p):

    count = 0

    while(True):

        I = alias_draw(J, q)
        
        #I = np.random.random_integers(0, N-1, size=1)
        count += 1        
        pest = p(x[I, :], y[I], t0, t1, theta, sigma)

        if np.random.random() <= pest:
            
            return np.array([I, count])


def bernoulli_resampling4(c, x0, x1, t0, t1, theta, sigma, p, pool):

    N = len(c)

    J, q = alias_setup(c)

    if pool is None:
        out = np.zeros(N)
        count = np.zeros(N)

        for i in range(N):

            out[i], count[i] = bernoulli_race4(J, q, N, x0, x1, t0, t1, theta, sigma, p)

        return out.astype(int), count

    else:
        out = pool.starmap(bernoulli_race4, [(N, x, t, theta, sigma, p)] * N)
        out = np.array(out)

        return out[:,0].astype(int), out[:,1]

# Resampling for Kalman Filter

def bernoulli_race_kf(N, x, y):

    count = 0
    #np.random.seed() # for parallel

    while(True):

        Us = np.random.rand(2)
        #I = alias_draw(J, q)
        I = np.random.random_integers(0, N-1, size=1)
        
        count += 1        

        X = 0.8*x[I] + np.sqrt(5)*np.random.standard_normal(1)

        if np.log(Us[1]) < (-(y - X)**2/(2*5)):

            return np.array([I, count])#I, count

def bernoulli_resampling_kf(N, x, y, pool):

    if pool is None:

        out   = np.zeros(N)
        count = np.zeros(N)

        for i in range(N):
            
            out[i], count[i] = bernoulli_race_kf(N, x, y)

        return out.astype(int), count

    else:

        out = pool.starmap(bernoulli_race_kf, [(N, x, y)] * N)
        out = np.array(out)

        return out[:,0], out[:,1] #

def bernoulli_race_kf2(N, x, y, A):

    dimension = A.shape[0]
    eye = np.eye(dimension)

    count = 0
    #np.random.seed() # for parallel

    while(True):

        U = np.random.rand(1)
        I = np.random.random_integers(0, N-1, size=1)
        
        count += 1    

        X = np.transpose(np.matmul(A, np.transpose(x[I, :]))) + np.sqrt(5)*np.random.standard_normal(dimension)

        dev = y - X[0]
        maha = np.sum(np.square(np.dot(dev, 1/5*eye)), axis=-1)

        if np.log(U) < - maha/2: #multivariate_normal.logpdf(y, X[0], 5*eye):

            return np.array([I, count])

def bernoulli_resampling_kf2(N, x, y, A, pool):

    if pool is None:

        out = np.zeros(N)
        count = np.zeros(N)

        for i in range(N):

            out[i], count[i] = bernoulli_race_kf2(N, x, y, A)

        return out.astype(int), count

    else:

        out = pool.starmap(bernoulli_race_kf2, [(N, x, y)] * N)
        out = np.array(out)

        return out[:,0], out[:,1] #

# Resampling for Nonlinear Filter

def bernoulli_race_nl(J, q, x, y, W, iteration):
    
    count = 0
    
    while(True):
        
        I = alias_draw(J, q)
        count += 1        

        X = x[I]/2 + 25*x[I]/(1+x[I]**2) + 8*np.cos(1.2*iteration) + np.sqrt(10)*np.random.randn(1)

        if np.log(np.random.rand(1)) < (-(y - X)**2/(2*10)):
            
            return I, count

def bernoulli_resampling_nl(c, x, y, W, iteration):
    
    N = len(c)
    out   = np.zeros(N)
    count = np.zeros(N)
    
    J, q = alias_setup(c)
    
    for i in range(N):
        
        out[i], count[i] = bernoulli_race_nl(J, q, x, y, W, iteration)
        
    return out.astype(int), count


if __name__=="__main__":
    
    from scipy.stats import binom, describe
    import pandas as pd
    N = 100

    w = binom.pmf(np.arange(N), 100, 0.5)

    m = np.zeros(N)
    v = np.zeros(N)

    from tqdm import tqdm

    for i in tqdm(range(100)):
            
        with Pool(processes=4) as p:

            I, C = bernoulli_resampling2(N, w, pool=p, parallel=True)

        #print(I)
        m[i] = np.mean(C)

        I, C = bernoulli_resampling2(N, w)
        v[i] = np.mean(C)
        #print(np.mean(C))
        #print(np.std(C))
        #print(describe(C))
    print(np.std(m))
    print(np.std(v))

    I2 = pd.Series(I)

    out = I2.value_counts().sort_index()/N
    print(out.index)
    print(pd.DataFrame(out).T)
    print(w[45:55])


    print(sum(w)/N)
    print((N-1)/(sum(C)-1))

    import matplotlib.pyplot as plt

    plt.hist(I, normed=True)
    plt.hist(w, color="red")
    plt.show()
