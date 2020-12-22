### Baisc heterogenous agent model

# Packages
import numpy as np
import matplotlib.pyplot as plt

# Parameters
rho = 0.06
sigma  = 2                      # inv. intertemporal el. of substitution              
beta   = 1/(1+rho)              # discount factor
s_size = 2                      # number of states
g_size = 45                     # number of generations
delta  = 0                      # depreciation rate
alpha  = 0.33                   # capital share
uopt   = 1                      # if 1 - CRRA, if 0 -quadratic

# shocks discretization
gamma = 0
std_dev = 0
shocks = np.array([1+std_dev, 1-std_dev])           # 1 - sigma, 1 + sigma
prob   = np.array([[(1+gamma)*0.5, (1-gamma)*0.5],
                   [(1-gamma)*0.5, (1+gamma)*0.5]]) # probabilities of changing state

# wage and interest rate
r    = 0.04
wage = 1

# Utility function - CRRA
def u_fun(c, mu, uoption):
    if uoption == 1:
        return (c**(1-mu))/(1-mu)
    elif uoption == 0:
        return -0.5*((c - 100)**2)       

##### OLG economy #############################################################

# form assets grids for different borrowing constraints
k_size  = 300                   # number of grid points
k_max   = 30
k_grid = np.zeros([g_size, k_size])
for n in range(0, 44):
    k_min   = -shocks[0]*(1+r)**(n-g_size+1)
    k       = np.linspace(k_min, k_max, k_size)
    k_grid[n, :] = k  
k_grid[44, :] = np.linspace(0, k_max, k_size)

### Backward induction: value function
# last generation
n = g_size - 1   
# Utility
utility = np.ones((s_size, k_size, k_size))*np.inf*(-1)
for q in range(0, s_size): #shock
    for i in range(0, k_size): #today
        for j in range(0, k_size): #tomorrow
            c_temp = wage*shocks[q] + k_grid[n, i]*(1+r) - k_grid[n, j]
            if c_temp > 0: #consumption must be non-negative
                utility[q,j,i] = u_fun(c_temp, sigma, uopt)

# Initalize some variables
value = np.zeros([s_size, g_size, k_size])
cont = np.zeros([s_size, g_size, k_size])
chi = np.zeros(( s_size, k_size, k_size))
v_temp = np.zeros([ k_size, s_size])

for s in range(0, s_size):
    for i in range(0, k_size): #today
        c_temp = wage*shocks[s] + k[i]*(1+r)
        if c_temp >= 0:
            value[s, n, i] = u_fun(c_temp, sigma, uopt)
        else:
            value[s, n, i] = np.inf*(-1)                

# rest of generations
for n in range(g_size-2, -1, -1):
    # Utility
    utility = np.ones((s_size, k_size, k_size))*np.inf*(-1)
    for q in range(0, s_size): #shock
        for i in range(0, k_size): #today
            for j in range(0, k_size): #tomorrow
                c_temp = wage*shocks[q] + k_grid[n, i]*(1+r) - k_grid[n, j]
                if c_temp > 0: #consumption must be non-negative
                    utility[q,j,i] = u_fun(c_temp, sigma, uopt)

    for q in range(0, s_size):
        v_temp[:, q] = prob[q,0]*value[0, n+1, :]+ prob[q,1]*value[1, n+1, :]
        chi[q, :, :] = utility[q,:,:] + beta*( np.matmul(v_temp[:,q].reshape(300,1), np.ones([ 1, k_size]))) 
        value[q, n, :] = np.max(chi[q,:,:], axis = 0)
        cont[q, n, :] = np.argmax(chi[q,:,:], axis = 0)

## Policy functions
# capital (asset) policy functions
k_prim = np.zeros([s_size, g_size, k_size])
cont  = cont.astype(int)
for n in range(0,45):
    for q in range(0,s_size):    
        for i in range(0,k_size):    
            k_prim[q, n, i] = k[cont[q, n, i]]

# consumption policy function
cons_pol = np.zeros([s_size, g_size, k_size])
for n in range(0,45):
    for q in range(0, s_size): #shock
        for i in range(0, k_size): #today
                cons_pol[q, n, i] = wage*shocks[q] + k[i]*(1+r) - k_prim[q, n, i]
                
## Plots
plt.plot(k_grid[5, :], cons_pol[0, n, :], label="high income")
plt.plot(k_grid[5, :], cons_pol[1, n, :], label="low income")
plt.title('Consumption policy function in economy with T=45.', fontsize = 14) # title with fontsize 20
plt.suptitle('Generation 5',fontsize=14)
plt.ylabel('Consumption', fontsize = 10) # x-axis label
plt.xlabel('Assets', fontsize = 10) # y-axis label
plt.legend()
plt.savefig('consumptionpol5.png')        
plt.show()

plt.plot(k_grid[40, :], cons_pol[0, n, :], label="high income")
plt.plot(k_grid[40, :], cons_pol[1, n, :], label="low income")        
plt.title('Consumption policy function in economy with T=45.', fontsize = 14) # title with fontsize 20
plt.suptitle('Generation 40',fontsize=14)
plt.ylabel('Consumption', fontsize = 10) # x-axis label
plt.xlabel('Assets', fontsize = 10) # y-axis label
plt.legend()
plt.savefig('consumptionpol40.png')        
plt.show()

## Simulated paths
# Choose arbitrary capital position and shock
a0 = np.int(np.round(k_size/2))
z0 = np.int(np.round(s_size/2))
    
# Initialize the simulation
T = 45      # the simulation horizon

# prepare vectors of consumption and assets
cons_simul = np.zeros(T)
assets_simul = np.zeros(T+1)
assets_index = np.zeros(T+1)
    
# fill the initial values
assets_index[0] = a0
assets_simul[0] = k_grid[0, z0]

assets_index[1] = cont[z0, 0, a0]
assets_simul[1] = k_grid[0, np.int(assets_index[1])]
cons_simul[0] = wage*shocks[z0] + (1+r)*assets_simul[0] - assets_simul[1]
    
# draw random shocks
rng = np.random.default_rng()
Z_t = rng.integers(s_size, size=g_size)

for t in range(0, g_size):
    a = cont[Z_t[t], t, np.int(assets_index[t])] 
    assets_index[t+1] = np.int(a)
    assets_simul[t+1] = k_grid[t, np.int(a)]
    cons_simul[t] = wage*shocks[np.int(Z_t[t])] + (1+r)*assets_simul[t] - assets_simul[t+1]    
    
plt.plot(np.linspace(0,T-1,T), cons_simul[0:45])
plt.title('Simulated Consumption path in economy with T=45.', fontsize = 14) # title with fontsize 20
plt.ylabel('Consumption', fontsize = 10) # x-axis label
plt.xlabel('Time', fontsize = 10) # y-axis label
plt.savefig('simulconsfinite.png')        
plt.show()
##### Infinitely-lived agent economy #########################################

# Finding fixed point for agregate capital stock
def excess_demand(r, dist_opt):
    # Forming capital grid
    k_size  = 300                   # number of grid points
    k_min   = -((1+r)/r)*shocks[0]*wage
    k_max   = 30
    k       = np.linspace(k_min, k_max, k_size)
    
    # capital and wage depending on rental rate
    mpk = 1 + r + delta
    N = 1
    K = N*((1/alpha)*mpk)**(1/(alpha-1))   

    # Utility
    utility = np.ones((s_size, k_size, k_size))*np.inf*(-1)
    for q in range(0, s_size): #shock
        for i in range(0, k_size): #today
            for j in range(0, k_size): #tomorrow
                c_temp = wage*shocks[q] + k[i]*(1+r) - k[j]
                if c_temp > 0: #consumption must be non-negative
                    utility[q,j,i] = u_fun(c_temp, sigma, uopt)
    
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
    excess = K - Kagg
    
    assidx = np.argsort(np.ndarray.flatten(k_prim))
    conidx = np.argsort(np.ndarray.flatten(c_pol))
    ass_dist = np.block([ [ probst[assidx] ], [ np.sort(np.ndarray.flatten(k_prim)) ] ])  
    con_dist = np.block([ [ probst[conidx] ], [ np.sort(np.ndarray.flatten(c_pol)) ] ])
    
    if dist_opt == 0:
        return excess
    elif dist_opt == 1:
        return  c_pol, k_prim, ass_dist, con_dist, cont, trans, probst, k

cons_pol, k_prim, ass_dist, con_dist, cont, trans, probst, k = excess_demand(r, 1)

## Plots
plt.plot( k_prim[0,:], cons_pol[0,:], label="high income")
plt.plot( k_prim[1,:], cons_pol[1,:], label="low income" )
plt.title('Consumption policy function in economy with T = infinite.', fontsize = 14) # title with fontsize 20
plt.ylabel('Consumption', fontsize = 10) # x-axis label
plt.xlabel('Assets', fontsize = 10) # y-axis label
plt.legend()
plt.savefig('consumptionpolinf.png')        
plt.show()

## Simulated paths
# Choose arbitrary capital position and shock
a0 = np.int(np.round(k_size/2))
z0 = np.int(np.round(s_size/2))
    
# Initialize the simulation
T = 45      # the simulation horizon

# prepare vectors of consumption and assets
cons_simul = np.zeros(T)
assets_simul = np.zeros(T+1)
assets_index = np.zeros(T+1)
    
# fill the initial values
assets_index[0] = a0
assets_simul[0] = k[0]

assets_index[1] = cont[a0, z0]
assets_simul[1] = k[np.int(assets_index[1])]
cons_simul[0] = wage*shocks[z0] + (1+r)*assets_simul[0] - assets_simul[1]
    
# draw random shocks
rng = np.random.default_rng()
Z_t = rng.integers(s_size, size=g_size)

for t in range(0, g_size):
    a = cont[np.int(assets_index[t]), Z_t[t]] 
    assets_index[t+1] = np.int(a)
    assets_simul[t+1] = k[np.int(a)]
    cons_simul[t] = wage*shocks[np.int(Z_t[t])] + (1+r)*assets_simul[t] - assets_simul[t+1]    
    
plt.plot(np.linspace(0,T-1,T), cons_simul[0:45])
plt.title('Simulated Consumption path in economy with T=infinite.', fontsize = 14) # title with fontsize 20
plt.ylabel('Consumption', fontsize = 10) # x-axis label
plt.xlabel('Time', fontsize = 10) # y-axis label
plt.savefig('simulconsinfinite.png')        
plt.show()