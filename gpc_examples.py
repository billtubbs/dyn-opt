# Compute Generalized Predictive Control (GPC) parameters
# for a discrete, single-input, single-output (SISO) system

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from polynomials import diophantine, diophantine_recursive


# Example 1
# System from Homework 5.1 with a process time delay of 2
# and also Exercise 6.2.

# Plant model (CARIMA)
A = [1, -0.8]
B = [0.4]
C = [1]
D = np.convolve(A,[1, -1])
d = 2  # process delay
lam = 0.1  # lambda parameter
hp = 4  # Length of prediction horizon
hc = 1  # Length of control horizon

# Solve the first Diophantine equation
F1, M1 = diophantine_recursive(C, D, hp)

# Since C = [1] we don't need to solve the 2nd 
# Diophantine equation

# G = F2  # TODO
Gc = F2[:,:hc]

# Solve GPC optimization problem
K.append(np.linalg.inv(Gc.T @ Gc + lam) @ Gc.T)
K = np.hstack(K)

# GPC gain is first row of K
K1 = K[1,:]

print(F1, M1)
print(F2, M2)
print(K)



# Example 2
# A simple example with prediction horizon of 1
# See Exercise 6.6 of GEL-7029 course

# Plant model (CARIMA)
A = [1, -0.9]
B = [-0.6]
C = [1, -0.7]
D = np.convolve(A,[1, -1])
d = 1  # process delay
lam = 0.1  # lambda parameter

# Solve the first Diophantine equation
F1, M1 = diophantine(C, D)

# Solve the 2nd Diophantine equation
F2, M2 = diophantine(B, C)  # F1 = 1 here

# Prediction matrices (Gc is a scalar here)
Gc = F2
# F = M2[0]*duf(k-1) + M1[0]*yf(k) + M1[1]*yf(k-1)
# where duf are future control commands, yf
# are filtered y values.

# GPC gain
K1 = Gc / (Gc**2 + lam)

# GPC control law (compute F each timestep):
# du = K1 @ (R - F)

print(F1, M1)
print(F2, M2)
print(Gc, K1)

assert F1 == 1.0
assert np.array_equal(M1, [1.2, -0.9])
assert F2 == -0.6 and Gc == F2
assert np.array_equal(M2, [-0.42])
assert np.isclose(K1, -1.3043478260869565)


# Test by simulating the closed-loop system with GPC controller
nT = 15
r0 = 1
k_ind = np.arange(-3, nT+2)  # Column vector

# Prepare arrays for simulation data
r = np.zeros_like(k_ind, dtype='float')
u = np.zeros_like(k_ind, dtype='float')
y = np.zeros_like(k_ind, dtype='float')
yf = np.zeros_like(k_ind, dtype='float')
du = np.zeros_like(k_ind, dtype='float')
duf = np.zeros_like(k_ind, dtype='float')
e = np.zeros_like(k_ind, dtype='float')
yhat = np.full_like(k_ind, np.nan, dtype='float')

# Random noise signal (rounded here)
#e[k_ind >= 0] = np.round(0.01 * np.random.randn(np.sum(k_ind >= 0)), 2)

# Integrate noise signal
e_cum = e.cumsum()

# Setpoint step
r[k_ind >= 0] = r0

for k in range(nT+1):  # k = 0 to nT
    i = k_ind.tolist().index(k)
    # System
    y[i] = -A[1]*y[i-1] + B[0]*u[i-1] + C[0]*e_cum[i] + C[1]*e_cum[i-1]
    # Filtered values
    yf[i] = -C[1]*yf[i-1] + y[i]
    # GPC control
    du[i] = 1.5652*yf[i] - 1.1739*yf[i-1] - 1.3043*r[i] - 0.5478*duf[i-1]
    u[i] = u[i-1] + du[i]
    # Filtered values
    duf[i] = -C[1]*duf[i-1] + du[i]
    # Prediction equation
    yhat[i+1] = F2*du[i] + M2[0]*duf[i-1] +  M1[0]*yf[i] + M1[1]*yf[i-1]

# Trim arrays to timesteps of interest
k_ind = np.array(k_ind)
i_ind = (k_ind >= 0) & (k_ind <= nT)
k_ind = k_ind[i_ind]
r = r[i_ind]
u = u[i_ind]
y = y[i_ind]
yf = yf[i_ind]
du = du[i_ind]
duf = duf[i_ind]
e = e[i_ind]
yhat = yhat[i_ind]

index = pd.Index(k_ind, name='k')
results = pd.DataFrame({
    'r': r,
    'u': u,
    'y': y,
    'yf': yf,
    'du': du,
    'duf': duf,
    'e': e,
    'yhat': yhat
}, index=index)

print(results)

# To plot in Matplotlib uncomment the following
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(2, 1)

# ax = axes[0]
# ax.plot(k_ind, r, label='r(k)')
# ax.plot(k_ind, y, marker='o', label='y(k)')
# ax.grid()
# ax.legend()
# ax.set_title('Reference and Output')

# ax = axes[1]
# ax.step(k_ind, u, where='post', label='u(k)')
# ax.grid()
# ax.set_xlabel('k')
# ax.legend()
# ax.set_title('Input')

# plt.show()

# Compare to results from MATLAB simulation
# See file ex6_6.mlx

du_test = np.array([
  -1.304300000000000,
   0.635089756000000,
   0.517573517608480,
   0.089520799680607,
  -0.064301937791926,
  -0.044079244992572,
  -0.005625015357046,
   0.006302209609840,
   0.003704120939149,
   0.000296905024104,
  -0.000602150279550,
  -0.000306802378804,
  -0.000008879000061,
   0.000056384482659,
   0.000025039544498,
  -0.000000682681808,
])
assert_array_almost_equal(du, du_test)

yhat_test = np.array([
    np.nan,
    0.782580000000000,
    1.105848146400000,
    1.086245367594912,
    1.014890386861968,
    0.989252066877475,
    0.992625125886974,
    0.999035888209750,
    1.001024248534345,
    1.000591300262990,
    1.000023503804310,
    0.999873777159227,
    0.999923104605935,
    0.999972826708009,
    0.999983745910280,
    0.999978549465624,
])
assert_array_almost_equal(yhat, yhat_test)




# Example 3
# More general solution method

# Plant model (CARIMA)
A = [1, -0.8]
B = [0.4]
C = [1]
D = np.convolve(A,[1, -1])
d = 2  # process delay
lam = 0.1  # lambda parameter
hp = 4  # Length of prediction horizon
hc = 1  # Length of control horizon

# Solve the first Diophantine equation
F1, M1 = diophantine_recursive(C, D, hp)

# Solve the 2nd Diophantine equation
F2 = []
M2 = []
K = []
for j in range(F1.shape[0]):
    F2_j, M2_j = diophantine_recursive(np.convolve(B, F1[j,0:j+1]), C, j+1)
    F2.append(F2_j[-1])  # Last row is final recursive result
    M2.append(M2_j[-1])

F2 = np.hstack(F2)
M2 = np.hstack(M2)

G = F2
Gc = F2[:,:hc]

# Solve GPC optimization problem
K.append(np.linalg.inv(Gc.T @ Gc + lam) @ Gc.T)
K = np.hstack(K)

# GPC gain is first row of K
K1 = K[1,:]

print(F1, M1)
print(F2, M2)
print(K)