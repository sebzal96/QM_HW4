### Baisc heterogenous agent model

# Packages
import numpy as np
import scipy.stats as stats
from scipy.sparse import csr_matrix
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from scipy import optimize

# Parameters
rho = 0.06
sigma  = 1                      # inv. intertemporal el. of substitution              
beta   = 0.9899 #1/(1+rho)      # discount factor
gamma = 0.9695
s_size = 2                      # number of states
std_dev = np.sqrt(0.0384)
shocks = np.array([1+std_dev, 1-std_dev])           # 1 - sigma, 1 + sigma
prob   = np.array([[(1+gamma)*0.5, (1-gamma)*0.5],
                   [(1-gamma)*0.5, (1+gamma)*0.5]]) # probabilities of changing state
delta  = 0.025                  # depreciation rate
A      = 1                      # production technology
alpha  = 0.36                   # capital share
N      = 1                     # labour 

# Utility function - CRRA
def u_fun(c,mu):
    return np.log(c)
    #    return (c**(1-mu))/(1-mu)

# Finding fixed point for agregate capital stock
def excess_demand(r, dist_opt):
    # capital and wage depending on rental rate
    mpk = r + delta
    K = N*((1/alpha)*mpk)**(1/(alpha-1))   
    wage = (1-alpha)*K**(alpha)*N**(-alpha)
    
    # Forming capital grid
    k_size  = 300                   # number of grid points
    k_min   = -(1/r)*shocks[0]*wage
    k_max   = 30
    k       = np.linspace(k_min, k_max, k_size)
        
    # Utility
    utility = np.ones((s_size, k_size, k_size))*np.inf*(-1)
    for q in range(0, s_size): #shock
        for i in range(0, k_size): #today
            for j in range(0, k_size): #tomorrow
                c_temp = wage*shocks[q] + k[i]*(1+r) - k[j]
                if c_temp > 0: #consumption must be non-negative
                    utility[q,j,i] = u_fun(c_temp, sigma)
    
    # Initializing some of variables
    value = np.zeros([k_size, s_size])          # value function
    cont  = np.zeros([k_size, s_size])          # decision rules
    v_temp = np.zeros([ k_size, s_size])
    chi = np.zeros(( s_size, k_size, k_size))   # subject for maximization
    progress = 10                               # change in value function
    
    # Iterating Bellman equation
    while progress > 10e-10:
        valueold = value
        value = np.zeros([ k_size, s_size])
        
        for q in range(0, s_size):
            v_temp[:,q] = prob[q,0]*valueold[:,0]+ prob[q,1]*valueold[:,1]
    
        for q in range(0, s_size):
            chi[q,:,:] = utility[q,:,:] + beta*( np.matmul(v_temp[:,q].reshape(k_size,1), np.ones([1, k_size]))) 
            value[:,q] = np.max(chi[q,:,:], axis = 0)
            cont[:, q] = np.argmax(chi[q,:,:], axis = 0)    
        diff = abs(value - valueold)
        progress = np.nanmax( np.max(diff, axis = 0), axis = 0)
    
    # Policy function for capital 
    k_prim = np.zeros([ s_size, k_size])
    c_pol = np.zeros([ s_size, k_size])
    cont  = cont.astype(int)
    for q in range(0,s_size):    
        for i in range(0,k_size):    
            k_prim[q,i] = k[cont[i,q]]
            c_pol[q,i] = wage*shocks[q] + k[i]*(1+r) - k_prim[q,i]

    # Forming transition matrix from state at t (row) to state at t+1 (column)
    G = np.zeros([s_size, k_size, k_size], dtype=int)
     
    for n in range(0, s_size):
        for i in range(0, k_size):
            G[n, i, cont[ i, n] ] = 1
    
    trans = np.block([ [G[0,:,:]*prob[0,0], G[0,:,:]*prob[0,1] ],
                       [G[1,:,:]*prob[1,0], G[1,:,:]*prob[1,1] ] ])
    
    probst = np.ones([k_size, s_size])*(1/(2*k_size))
    probst = probst.flatten('F')
    
    progress = 1
    while progress > 1e-10:
          probst1 = np.matmul(trans.transpose(),probst)
          progress = max(abs(probst1-probst))
          probst = probst1
    
    Kagg = np.matmul(probst.transpose(), k_prim.flatten())
    c_pol = c_pol.flatten()
    k_prim = k_prim.flatten()
    excess = K - Kagg
    
    assidx = np.argsort(np.ndarray.flatten(k_prim))
    conidx = np.argsort(np.ndarray.flatten(c_pol))
    ass_dist = np.block([ [ probst[assidx] ], [ np.sort(np.ndarray.flatten(k_prim)) ] ])  
    con_dist = np.block([ [ probst[conidx] ], [ np.sort(np.ndarray.flatten(c_pol)) ] ])
    #ass_dist = np.block([ [ probstsort ], [ k_prim[probstind] ] ]) 
    #probd = np.sum(np.block([ [probst[0:300]], [probst[300:600]] ]), 0)
    
    if dist_opt == 0:
        return excess
    elif dist_opt == 1:
        return  c_pol, k_prim, ass_dist, con_dist, cont, trans, probst

sol = optimize.root_scalar(excess_demand, bracket=[-delta + 0.00000001, 10], args=(0), method='brentq')
r_eq = sol.root

c_pol, k_prim, ass_dist, con_dist, cont, trans, probst = excess_demand(r_eq, 1)

plt.plot(  con_dist[1,:], con_dist[0,:])
plt.title('Consumption distribution.', fontsize = 18) # title with fontsize 20
plt.ylabel('Mass', fontsize = 10) # x-axis label with fontsize 15
plt.xlabel('Consumption', fontsize = 10) # y-axis label with fontsize 15
plt.savefig('consdistribution.png')
plt.show()

plt.plot(ass_dist[1,:], ass_dist[0,:])
plt.title('Assets distribution.', fontsize = 18) # title with fontsize 20
plt.ylabel('Mass', fontsize = 10) # x-axis label with fontsize 15
plt.xlabel('Assets', fontsize = 10) # y-axis label with fontsize 15
plt.savefig('assetsdistribution.png')
plt.show()
