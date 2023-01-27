# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:01:35 2022

Author: Adarsh Ravindran

Title: Takens Reconstructions of the Lorenz Attractor

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.dpi'] = 300 
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams['legend.fontsize'] = 10

################# Part 1: Lorenz system generation #################

def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
     
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
     
    return [dx, dy, dz]
 
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
 
para = (sigma, beta, rho)  # Parameters of the system
y0 = [-8.0, 8.0, 27.0]  # Initial state of the system

t_span = (0.0, 200.0)
t = np.arange(0.001, 200.001, 0.001)
 
LorenzSystem = solve_ivp(lorenz, t_span, y0, args=para, method='BDF', t_eval=t) # Using BDF

fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.gca(projection='3d')
ax.plot(LorenzSystem.y[0, :],
        LorenzSystem.y[1, :],
        LorenzSystem.y[2, :])
ax.view_init(elev=12, azim=-60)
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$y$', fontsize=15)
ax.zaxis.set_rotate_label(False) 
ax.set_zticks([5, 10, 15, 20, 25, 30, 35, 40, 45])
ax.set_zlabel('$z$', fontsize=15, rotation = 0)
ax.set_title("Lorenz System", fontsize=35)

# Using BDF, different solver to ODE45 which was used in MATLAB. Should not make a major difference.

################# Part 2: Takens reconstructions #################

# X -reconstruction
fig = plt.figure()
fig.set_tight_layout(True)
X = LorenzSystem.y[0,:];
Y = LorenzSystem.y[1,:];
Z = LorenzSystem.y[2,:];
plt.plot(LorenzSystem.y[0, :])

tau=50; # time delay between observation functions

# Observation functions. 3 required for the Lorenz attractor.
a= X[:-2*tau];
b= X[tau:-tau];
c= X[2*tau:]

# Plot Takens Delay Time Embedding
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.gca(projection="3d")
ax.plot(xs=a, ys=b, zs=c, linewidth=0.25, color="navy", alpha=0.85)
ax.view_init(elev=30, azim=-48)
ax.set_title("Taken´s Embedding Approach with tau= " + str(tau))

# Same can be done for Y,Z coordinates. Behaviour is similar to that shown in part 3.
# Hankel delay embeddings give better reconstructions. See Broomhead 1986.

################# Part 3: Hankel delay embedding reconstructions #################

# In this section, we reconstruct the X,Y and Z shadow attractors using a Hankel matrix delay embedding.

q=100; # Number of stack shifted rows, cols of Hankel Matrix

### X time series reconstruction ###
X = LorenzSystem.y[0,:];

# Hankel matrix
H=np.empty((q,np.size(X)-q));
k=0;
while k<q:
    H[k,:]=X[k:np.size(X)-q+k]
    k=k+1

# SVD of Hankel matrix
U, S, Vt = np.linalg.svd(H, full_matrices=False)
V=np.transpose(Vt)

# Shadow attractor
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.gca(projection='3d')
ax.plot(V[:, 0],
        V[:, 1],
        V[:, 2])
ax.view_init(elev=30, azim=-78)
ax.xaxis.set_rotate_label(False) 
ax.set_xlabel('$v_1$', fontsize=15)
ax.yaxis.set_rotate_label(False) 
ax.set_ylabel('$v_2$', fontsize=15)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$v_3$', fontsize=15, rotation = 0)
ax.set_title("X-shadow attractor", fontsize=35)
plt.setp(ax.get_xticklabels(), visible=False);
plt.setp(ax.get_yticklabels(), visible=False);
plt.setp(ax.get_zticklabels(), visible=False);

### Y time series reconstruction ###
Y = LorenzSystem.y[1,:];

# Hankel matrix
H_Y=np.empty((q,np.size(Y)-q));
k=0;
while k<q:
    H_Y[k,:]=Y[k:np.size(Y)-q+k]
    k=k+1

# SVD of Hankel matrix
U_Y, S_Y, Vt_Y = np.linalg.svd(H_Y, full_matrices=False)
V_Y=np.transpose(Vt_Y)

# Shadow attractor
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.gca(projection='3d')
ax.plot(V_Y[:, 0],
        V_Y[:, 1],
        V_Y[:, 2])
ax.view_init(elev=55.112740957861583, azim=1.196823554789552e+02) # selected after trial and error.
ax.xaxis.set_rotate_label(False) 
ax.set_xlabel('$v_1$', fontsize=15)
ax.yaxis.set_rotate_label(False) 
ax.set_ylabel('$v_2$', fontsize=15)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$v_3$', fontsize=15, rotation = 0)
ax.set_title("Y-shadow attractor", fontsize=35)
plt.setp(ax.get_xticklabels(), visible=False);
plt.setp(ax.get_yticklabels(), visible=False);
plt.setp(ax.get_zticklabels(), visible=False);

### Z time series reconstruction
Z = LorenzSystem.y[2,:];

# Hankel matrix
H_Z=np.empty((q,np.size(Z)-q));
k=0;
while k<q:
    H_Z[k,:]=Z[k:np.size(Z)-q+k]
    k=k+1

# SVD of Hankel matrix
U_Z, S_Z, Vt_Z = np.linalg.svd(H_Z, full_matrices=False)
V_Z=np.transpose(Vt_Z)

# Shadow attractor
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.gca(projection='3d')
ax.plot(V_Z[:, 0],
        V_Z[:, 1],
        V_Z[:, 2])
ax.view_init(elev=50, azim=30)
ax.xaxis.set_rotate_label(False) 
ax.set_xlabel('$v_1$', fontsize=15)
ax.yaxis.set_rotate_label(False) 
ax.set_ylabel('$v_2$', fontsize=15)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$v_3$', fontsize=15, rotation = 0)
ax.set_title("Z-shadow attractor", fontsize=35)
plt.setp(ax.get_xticklabels(), visible=False);
plt.setp(ax.get_yticklabels(), visible=False);
plt.setp(ax.get_zticklabels(), visible=False);

