# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:20:41 2023

@author: alexb
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
a=1
b=2
c=1
d=0.3

# System of ODEs
def system(y, t):
    z1,z2 = y
    dz1dt = z1*(a-b*z2)
    dz2dt = z2*(c*z1-d)
    return [dz1dt,dz2dt]

# Initial conditions
z10 = 0.2
z20 = 0.3

# Time vector
t = np.linspace(0, 13, 100)

# Solve ODE
solution = odeint(system, [z10,z20], t)

# Extract v and w
z1 = solution[:, 0]
z2 = solution[:, 1]

# Store results in DataFrame
data = pd.DataFrame({
    't': t,
    'z1': z1,
    'z2': z2,
})

# Print DataFrame
print(data)

# Save DataFrame to CSV
data.to_csv('synthetic_data_LK.csv', index=True)

