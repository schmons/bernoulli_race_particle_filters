import numpy as np

def multinomial_sampling(w):
    
    N = len(w)
    
    U = create_uniform(N)
    #U = sorted(np.random.rand(N))
    W = np.append(0, np.cumsum(w))
    #W.append(1)
    
    # define new index
    index = np.zeros(N)
    
    I = 0
    J = 0

    for i in range(N):
        
        while(True):
            if W[J] <= U[i] and U[i] < W[J+1]:
                index[i] = J
                break
            else:
                J += 1

    return index.astype(int)

def create_uniform(N):
    
    U = np.random.rand(N)
    U[N-1] = U[N-1]**(1/N)
    #print(len(U))
    #U[N-1] = np.random.rand(1)
    #V      = np.random.rand(N-1)
    
    for i in range(1, N):
        #print(N-1-i)
        #V = np.random.rand(1)
        U[N-1-i] = U[N-i]*(U[N-1-i]**(1/(N-i)))
        #print(U[N-1-i])
    return U

if __name__=="__main__":

    import matplotlib.pyplot as plt 
    
    C =  create_uniform(10)
    w = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    W = multinomial_sampling(w/sum(w))

    for i in range(10000):
    
        C = np.append(C, create_uniform(10))
        W = np.append(W, multinomial_sampling(w/sum(w)))

    plt.hist(C)
    plt.hist(W)
    plt.show()