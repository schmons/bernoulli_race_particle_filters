import numpy as np
import bisect

from tqdm import tqdm

# General SDE stuff

def brownian_bridge_unscaled(t_grid, T):

    t_grid = np.r_[t_grid, T]

    N = len(t_grid)
    #Z = np.random.randn(N)
    W = brownian_motion(t_grid)

    BB = np.zeros(N)

    for i in range(N):

        BB[i] = W[i] - t_grid[i]/T*W[N-1]
        #W0*(1 - t_grid[i]/T) + WT*t_grid[i]/T + (W[i] - t_grid[i]/T*W[N-1])  

    return(BB)

def brownian_bridge(W0, WT, t_grid, T):

    t_grid = np.r_[t_grid, T]

    N = len(t_grid)
    #Z = np.random.randn(N)
    W = brownian_motion(t_grid)

    BB = np.zeros(N)

    for i in range(N):

        BB[i] = W0*(1 - t_grid[i]/T) + WT*t_grid[i]/T + (W[i] - t_grid[i]/T*W[N-1])  

    return(BB)

def brownian_motion(t_grid):
    
    N = np.array(t_grid).size
    
    Z = np.random.standard_normal(N)
    W = np.zeros(N)
    
    for i in range(0, N):
        
        if i == 0:
            W[i] = np.sqrt(t_grid[i])*Z[0]
        else:
            W[i] = W[i-1] + np.sqrt(t_grid[i]-t_grid[i-1])*Z[i]
        
    return(W)


# Stuff for sine diffusion
#
#def random_h(T):
#    
#    while(True):
#        
#        U = np.random.rand(1)
#        V = np.sqrt(T)*np.random.standard_normal(1)
#        
#        if(np.log(U) <= -np.cos(V) - 1):
#                  
#            return(V)  
#
def g_function(u):
    
    y = (np.sin(u)**2 + np.cos(u)+1)/2
    
    return(y)

def phi_function(u):
    
    y = np.sin(u)**2/2 + np.cos(u)/2
    
    return(y)

# Rejection step for Exact Algorithm

def exact_sampling(start, T):
    
    W0 = start
    WT = random_h2(start, T)
    
    SDE_out = np.array([W0, WT])
    t_out   = np.array([0., T])
    
    U = np.random.rand(1)
    
    i = 0
    
    while(True):
        
        V   = T*np.random.rand(1)
        W   = np.random.rand(1)/T
        i   += 1
        B_V = np.sqrt(V)*np.random.randn(1)
        
        right = bisect.bisect_right(t_out, V)
        left = right - 1
              
        BB_V = SDE_out[left]*(1 - V/T) + SDE_out[right]*V/T + (B_V - V/T*SDE_out[right])
        #print(BB_V)
        SDE_out = np.insert(SDE_out, right, BB_V)
        #print(SDE_out)
        t_out   = np.insert(t_out, right, np.array([V]))
        #print(t_out)
        if(g_function(BB_V) <= W or U >= 1/np.math.factorial(i)):
        
            if(i % 2 == 0):
                I = 0
                return(SDE_out, t_out, I)
            else: 
                I = 1
                return(SDE_out, t_out, I)


def exact_sampler(start, T):
    
    while(True):
        
        S, t_out, I = exact_sampling(start, T)
        
        if(I == 1):
        
            return(S, t_out)

# Restrospective Sampler

def random_h2(start, T):
    
    while(True):
        
        U = np.random.rand(1)
        V = start + np.sqrt(T)*np.random.standard_normal(1)
        
        if np.log(U) <= (1-np.cos(V) - 2):
                  
            return(V) 

# def sine_ssm_alt(x0, T, noise):
    
#     t, sde = euler_sin_unbiased(x0, 0.25, 2*T+1)
    
#     Y = np.zeros(T+1)
#     t_grid = np.zeros(T+1)
#     S = np.zeros(T+1)
#     j = 0
    
#     for i in range(len(sde)):
#         if i % 2 == 0:
#             Y[j] = sde[i] + noise*np.random.randn()
#             t_grid[j] = t[i]
#             S[j] = sde[i]
#             j += 1
    
#     return Y, S, t_grid

# All the coin flips and estimators

def poisson_estimator(x, y, t, la):
    
    kappa = np.random.poisson(lam=la*t, size=1)
    c = 5/8 

    if kappa == 0:
        
        PE = 1#np.exp((la-c)*t)
        return PE

    phis = np.random.uniform(0, t, size=kappa)

    t_grid = np.sort(phis)

    BB = brownian_bridge(x, y, t_grid, t)

    BB = BB[:-1]

    PE = 1*la**(-kappa)*np.prod(c - phi_function(BB))#*np.exp((la-c)*t)

    return PE

def poisson_estimator2(x, y, t, la):

    k = np.random.poisson(lam=la*t, size=1)

    c = 5/8

    if k == 0:

        PE = 1#np.exp((la-c)*t)
        return PE

    p  = np.sort(np.random.uniform(0, t, size=k))

    Z  = np.random.standard_normal(k)
    W  = np.zeros(k)
    BB = np.zeros(k)

    W[0] = np.sqrt(p[0])*Z[0]

    for j in range(1, int(k)):
        
        W[j]  = W[j-1] + np.sqrt(p[j]-p[j-1])*Z[j]
        
    WT = np.sqrt(t - p[-1])*W[-1]
                
    BB = W - p/t*WT
    
    PE = 1*la**(-k)*np.prod(c - phi_function(BB + (1 - p/t)*x + p/t*y))#*np.exp((la-c)*t)
    
    return PE


def poisson_pgf_coin(x, y, t, la):
    
    k = np.random.poisson(lam=la*t, size=1)
    p = np.zeros(k)
    c = 5/8
    
    if k == 0:

        return 1

    U = np.sort(np.random.uniform(0, t, size=k))
    U = np.insert(U, 0, 0)
    U_diff = U[1:len(U)] - U[0:(len(U)-1)]

    U = U[1:]

    BI = np.sqrt(U_diff)*np.random.standard_normal(k)
    #print(BI)
    B = np.cumsum(BI)
    
    BB = np.zeros(k)

    for i in range(int(k)):
        
        BB[i] = B[i] + (1 - U[i]/t)*x + (U[i]/t)*y
        
    for i in range(int(k)):

        #W    = np.random.normal(loc=U, scale=np.sqrt(t), size=1)
        #print(phi_function(BB[i]))
        p[i] = np.random.random() <= (c - phi_function(BB[i]))/la
        #p[i] = np.random.binomial(size=1, n=1, p=(c - phi_function(BB[i]))/la)      

    return np.prod(p)

def exact_coin(start, end, T, la):

    W0 = start
    WT = end

    SDE_out = np.array([W0, WT])
    t_out = np.array([0., T])

    U = np.random.rand(1)

    i = 0

    while(True):

        V   = T*np.random.rand(1)
        W   = np.random.rand(1)/T
        i   += 1
        B_V = np.sqrt(V)*np.random.randn(1)
        
        right = bisect.bisect_right(t_out, V)
        left = right - 1
              
        BB_V = SDE_out[left]*(1 - V/T) + SDE_out[right]*V/T + (B_V - V/T*SDE_out[right])
        #print(BB_V)
        SDE_out = np.insert(SDE_out, right, BB_V)
        #print(SDE_out)
        t_out   = np.insert(t_out, right, np.array([V]))
        #print(t_out)
        if(g_function(BB_V) <= W or U >= 1/np.math.factorial(i)):

            if(i % 2 == 0):
                I = 0
                return(I)
            else: 
                I = 1
                return(I)


def retrospective_sampling(start, T):

    k = np.random.poisson(lam=T, size=1)

    r = 9/8

    u = np.random.uniform(0, r, size=k)
    x = sorted(np.random.uniform(0, T, size=k))
    x = np.insert(x, 0, 0)

    rho = random_h2(start,T)

    BB  = brownian_bridge(start, rho, x, T)

    for j in range(int(k)):
        
        if u[j] <= g_function(BB[j+1]):

            return x, BB, 0

    return np.append(x, T), np.append(BB, rho), 1
        

def retrospective_sampler(start, T):
    
    while(True):
        
        t_out, S, I = retrospective_sampling(start, T)
        
        if(I == 1):
        
            return(S, t_out)


def exp_coin(x, y, T, la=9/8):

    W0 = x
    WT = y

    n = 0
    U = 1
    L = 0
    
    G = np.random.rand(1)
  
    X = 0
  
    while(True):
        
        if(G <= L):

            return(1)
    
        if(G >= U):

            return(0)
        n = n + 1
    
    if(n % 2 == 0):
      
      X = c(X, p)
      U = L + np.prod(X)/np.math.factorial(n)
      
    else:
      
      X = c(X, p)
      L = U - np.prod(X)/np.math.factorial(n)


def retrospective_coin2(start, end, T):
    
    x = np.array([0])
    k = 0
    r = 9/8
    while x[-1] < T:
        
        x = np.append(x, np.random.exponential(scale=1/r, size=1))
        k += 1
        
    rho = end

    BB  = brownian_bridge(start, rho, x, T)

    BB = BB[:-1]
    u = np.random.uniform(0, r, size=k)

    for j in range(int(k)):
        
        if u[j] <= g_function(BB[j+1]):

            return 0

    return 1


# Sde sims

def rejection_sampler(X_old, delta):
    
    while(True):
        
        X_new = X_old + np.sqrt(delta)*np.random.standard_normal(1)
    
        U = np.random.rand()
        
        #pe = exact_coin(X_old, X_new, delta, la=9/8)
        pe = poisson_pgf_coin(X_old, X_new, delta, la=9/8)
        
        #A  = -np.cos(X_new) -1 - 1/2*(delta) #
        A  = -np.cos(X_new)+np.cos(X_old)-2# + np.log(pe) #why did through this out?
        
        if(np.log(U) <= A and pe):
            
            return(X_new)

def euler_sin_unbiased(S0, eps, T):
    
    t = np.arange(0, T*eps, eps)
    N = len(t) # length
    S = np.zeros(N)
    S[0] = S0
    
    W = np.random.standard_normal(N) # genrate standard normals
    
    for i in (range(1, N)):
        
        S[i] = rejection_sampler(S[i-1], eps)
        
    return(t, S)

def sine_ssm_alt(x0, T, noise):
    
    t, sde = euler_sin_unbiased(x0, 0.25, 2*T+1)
    
    Y = np.zeros(T+1)
    t_grid = np.zeros(T+1)
    S = np.zeros(T+1)
    j = 0
    
    for i in range(len(sde)):
        if i % 2 == 0:
            Y[j] = sde[i] + noise*np.random.randn()
            t_grid[j] = t[i]
            S[j] = sde[i]
            j += 1
    
    return Y, S, t_grid


if __name__ == "__main__":

    N = 50000

    pe_est = np.zeros(N)
    pe_est2 = np.zeros(N)
    exact_est1 = np.zeros(N)
    exact_est2 = np.zeros(N)
    exact_est3 = np.zeros(N)

    start = np.pi
    end = 0

    for i in tqdm(range(N), ascii=True, ncols=100):
        
        pe_est[i] = poisson_estimator(start, end, t=0.5, la=9/8)
        pe_est2[i] = poisson_estimator2(start, end, t=0.5, la=9/8)
        exact_est1[i] = poisson_pgf_coin(start, end, 0.5, la=9/8)
        exact_est2[i] = exact_coin(start, end, 0.5, la=9/8)
        #exact_est3[i]  = retrospective_coin2(start, end, 0.5)


    t  = 0.5
    la = 9/8
    c  = 5/8

    print(np.mean(pe_est))
    print(np.var(pe_est))
    print(np.mean(pe_est2))
    print(np.var(pe_est2))
    print(np.mean(exact_est1))#np.exp(la*t-c*t)*
    print(np.var(exact_est1))
    print(np.mean(exact_est2))
    print(np.var(exact_est2))
    #print(np.mean(exact_est3))
    #print(np.var(exact_est3))