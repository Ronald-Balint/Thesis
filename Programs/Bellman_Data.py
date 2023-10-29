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
k1 = 0.00000457
k2 = 0.00027845

# System of ODEs
def system(y, t):
    z = y
    dzdt = k1*(126.2-z)*(91.9-z)**2-k2*z**2
    return dzdt

# Initial conditions
z0 = 0

# Time vector
t = np.linspace(0, 50, 101)

# Solve ODE
solution = odeint(system, z0, t)

# Extract v and w
z = solution[:, 0]


# Store results in DataFrame
data = pd.DataFrame({
    't': t,
    'z': z,

})

# Print DataFrame
print(data)

# Save DataFrame to CSV
data.to_csv('synthetic_data_Bellman.csv', index=True)

