import numpy as np
from math import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#============================================FUNCTION ISING========================================
def ising(numiter, N, T, H):
    "the ising function"
    J  = 1.
    k  = 1.
    mu = 1.

    burn_in = 50000
    assert numiter > burn_in

    #generate empty arrays for storing Momentum M and energy E
    M  = np.array([])
    E  = np.array([])
    M2 = np.array([])
    E2 = np.array([])

    # Generate initial spins matrix
    grid = np.zeros(shape=(N, N))
    grid = grid.astype(float)

    grid = (grid - 0.5) * 2
    grid = grid.astype(int)


    for i in range(numiter+1):

        x, y = np.random.randint(0, N, size=2)
        flip_spin = False

        # Rolled arrays for neighbor calculation
        x1 = np.roll(grid, +1, axis=1)
        x2 = np.roll(grid, -1, axis=1)
        x3 = np.roll(grid, +1, axis=0)
        x4 = np.roll(grid, -1, axis=0)

        dx = x1 + x2 + x3 + x4

        dE = 2. * J * grid[x, y] * dx[x, y] + 2. * H * grid[x, y]

        if dE <= 0: flip_spin = True
        elif np.random.random() < np.exp(-dE/k/T): 
                    flip_spin = True

        if flip_spin: grid[x, y] *= -1

        currM = np.sum(mu * grid)
        currE = np.sum(-0.5 * 0.5 * J * grid * dx - H * grid)

        currM2 = np.sum(mu * mu * grid * grid)
        currE2 = np.sum((-0.5 * 0.5 * J * grid * dx - H * grid) ** 2)
 
        M  = np.append(M,  currM  / (N * N))
        E  = np.append(E,  currE  / (N * N))
        M2 = np.append(M2, currM2 / (N * N))
        E2 = np.append(E2, currE2 / (N * N))

    Mmean  = np.mean(M [burn_in :])
    Emean  = np.mean(E [burn_in :])
    Mmean2 = np.mean(M2[burn_in :])
    Emean2 = np.mean(E2[burn_in :])

    return Mmean, Emean, Mmean2, Emean2

#Mmean: mean value of the magnetic momentum (per site), over all the (converged) simulation time
#Mmean2: mean value of the square of the magnetic momentum (per site), over all the (converged) simulation time
#Emean: mean value of the energy (per site),  over all the (converged) simulation time
#Emean2: mean value of the square of the energy (per site), over all the (converged) simulation time
#===============================================================================

#Initial Configuration
n_grid  = 50
numiter = 200000
tgrid   = range(1, numiter + 1)

Ms  = np.array([])
Ts  = np.array([])
Es  = np.array([])
chi = np.array([])
cv  = np.array([])
Hs  = np.array([])

#record the data for 20 different values of the temperature---------

for i in range(1, 26):
    H = 0.
    T = 0.1 + 0.16 * (i-1)

    [Mmean, Emean, Mmean2, Emean2] = ising(numiter, n_grid, T, H)

    mychi = 1 / T    * (Mmean2 - Mmean * Mmean)
    mycv  = 1 / T**2 * (Emean2 - Emean * Emean)

    Ms  = np.append(Ms,  abs(Mmean))
    Es  = np.append(Es,  Emean)
    chi = np.append(chi, mychi)
    cv  = np.append(cv,  mycv)
    Ts  = np.append(Ts,  T)
    

plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.plot(Ts[8:20], Es[8:20], 'ro', linestyle = '--')
plt.ylabel('energy per site')
plt.xlabel('Temperature')

plt.subplot(222)
plt.plot(Ts[8:20], Ms[8:20], 'bo', linestyle = '--')
plt.ylabel('magnetization per site')
plt.xlabel('Temperature')

plt.subplot(223)
plt.plot(Ts[8:20], chi[8:20], 'go', label='Data', linestyle = '--')
plt.axhline(0, color='black', linewidth=0.5)  # Add this line to draw a horizontal line at y=0
plt.ylabel('magnetization susceptibility per site', fontsize = 'small')
plt.xlabel('Temperature')

plt.subplot(224)
plt.plot(Ts[8:20], cv[8:20], 'mo', linestyle = '--')
plt.ylabel('specific heat per site')
plt.xlabel('Temperature')

plt.tight_layout()
plt.savefig("results/ex2019.pdf") # saving the result as a pdf

plt.show()

myrestable = np.vstack((Ts, Es, chi, cv)).T
np.savetxt("results/output2019.dat", myrestable, delimiter = "\t")
