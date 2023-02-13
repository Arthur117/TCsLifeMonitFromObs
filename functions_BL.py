from tqdm import tqdm
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import copy

### FUNCTIONS
### BOUNDARY LAYER MODEL
### FOLLOWING SMITH AND MONTGOMERY 2020

def sm20(r, nu1, sm, rm, x):
    '''
    Just for test.
    Function as defined in SM2020.
    To reproduce their graphical results in appendix use
    v16 = sm20(rs, 97, 2, 50000, x=1.6)
    v23 = sm20(rs, 99, 1.3, 50000, x=2.3)
    '''
    s   = sm * r / rm
    return nu1 * s / (1 + s ** x)

def I(r, v, fcor):
    '''Inertial stability'''
    dr   = np.unique(np.diff(r))[0] # meters
    ksi  = 2 * v / r + fcor
    zeta = np.gradient(v, dr) + v / r + fcor
    return np.sqrt(ksi * zeta)

def delta(K, i):
    '''BL height scale'''
    return np.sqrt(2 * K / i)

def nu(v, Cd, delta, K):
    '''Parameter nu of SM 2020'''
    return Cd * v * delta / K

def a1(nu):
    '''Coeff a1 of SM2020, given by the corrected version'''
    num = - nu * (nu + 1)
    den = 2 * nu ** 2 + 3 * nu + 2
    return (num / den)

def a2(nu):
    '''Coeff a2 of SM2020, given by the corrected version'''
    num = nu
    den = 2 * nu ** 2 + 3 * nu + 2
    return (num / den)

def v_prime(r, v, fcor, Cd, K):
    '''Expression of v prime in SM2020, to get the total tangential wind add v_gradient''' 
    i = I(r, v, fcor)
    d = delta(K, i)
    n = nu(v, Cd, d, K)
    return v * a1(n)

def ki(r, v, fcor):
    '''Coefficient ki as defined in SM2020'''
    dr   = np.unique(np.diff(r))[0] # meters
    ksi  = 2 * v / r + fcor
    zeta = np.gradient(v, dr) + v / r + fcor
    return np.sqrt(ksi / zeta)

def u(r, v, fcor, Cd, K):
    '''Expression of the (total) radial wind speed in SM2020'''
    i = I(r, v, fcor)
    d = delta(K, i)
    n = nu(v, Cd, d, K)
    return - ki(r, v, fcor) * v * a2(n)


### FOLLOWING ELIASSEN AND LYSTAD 1977

def omega(r, omg_abs, chi, t, Cd=7e-2, H_h=1e4):
    num = omg_abs * (H_h)
    den = H_h + omg_abs * (chi ** 2) * Cd * r * t
    return num / den

def v_evolved(r, v, fcor, chi, t, Cd=7e-2, H_h=1e4):
    omg   = v / r - fcor
    omg_t = omega(r, abs(omg), chi, t, Cd, H_h)
    return (fcor + np.sign(omg) * omg_t) * r



