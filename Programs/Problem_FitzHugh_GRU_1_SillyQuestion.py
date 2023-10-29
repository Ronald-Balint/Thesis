# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 09:35:46 2023

@author: alexb
"""

from tensorflow import keras as K
from tensorflow import math
import tensorflow as tf
from tensorflow import reduce_sum
from tensorflow import math
from tensorflow import square
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from findiff import FinDiff
import numpy as np
from scipy.interpolate import CubicSpline
import csv
import sys
from scipy.integrate import odeint
import pandas as pd

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

#Master Reset
K.backend.clear_session()


# Open the file in 'r' mode, not 'rb'
csv_file = open('synthetic_data.csv','r')
index = []
t = []
v =[]
w=[]

# Read off and discard first line, to skip headers
csv_file.readline()

# Split columns while reading
for a, b,c,d in csv.reader(csv_file, delimiter=','):
    # Append each variable to a separate list
    index.append(float(a))
    t.append(float(b))
    v.append(float(c))
    w.append(float(d))

#Re-Label columns and group as needed       
train=np.column_stack((index,t))
train_data=train.tolist()
v=tf.cast(v,tf.float32)
w=tf.cast(w,tf.float32)

#Grouping columns and changing to list form
this_wont_work=np.column_stack((v,w))
again = this_wont_work.tolist()

#Structure of Network
model0 = K.Sequential([
    layers.Dense(80, activation="tanh"),
    layers.Dense(80, activation="tanh"),
    layers.Dense(80, activation="tanh"),
    layers.Dense(80, activation="tanh"),
    layers.Dense(80, activation="tanh"),
    layers.Dense(80, activation="tanh"),
    layers.Dense(2, activation='linear')
    ])

# =============================================================================
# bestgraphmodel = K.callbacks.ModelCheckpoint(
#     "C:\Anaconda3\FHN_RNN_Graph.keras",
#     monitor = "loss",
#     verbose = 0,
#     save_best_only = True,
#     save_freq="epoch")
# =============================================================================

model0.compile(
    optimizer="adam",
    loss="mse"
    )

model0.fit(train_data, 
           again, 
           epochs=1,
           verbose=2,
           #callbacks=bestgraphmodel
          )
model0.load_weights("C:\Anaconda3\FHN_RNN_Graph.keras")
predict0=model0.predict(train_data)

#Plotting Estimates
plt.figure()
plt.plot(t, v, "bo", label="v")
plt.plot(t, w, "go", label="w")
plt.plot(t,predict0[:,0],"green", label="v NN Prediction")
plt.plot(t,predict0[:,1],"red", label="w NN Prediction")
plt.legend()


#Calculating Numerical Derivatives

delta = t[1]-t[0]

d_dt = FinDiff(0,delta,1)
dv_dt = tf.cast(d_dt(predict0[:,0]),tf.float32)
dw_dt = tf.cast(d_dt(predict0[:,1]),tf.float32)

#Setting up NN inputs and Outputs for Parameter Estimation    
vR=tf.cast(predict0[:,0],tf.float32)
wR=tf.cast(predict0[:,1],tf.float32)


#Defining Custom Loss Function  
def my_loss_fn(y_true, y_pred):
    # System of ODEs
    a=float(y_pred[0,0])
    b=float(y_pred[0,1])
    c=float(y_pred[0,2])
    
    def system(y, t):
        v1, w1 = y
        dvdt = (1/c)*(v1-(1/3)*v1**3-w1)
        dwdt = c * (v1 - a*w1 + b)
        derp = [dvdt, dwdt]
        return derp

    # Initial conditions
    v0 = -1
    w0 = 1

    # Solve ODE
    solution = odeint(system, [v0, w0], t)

    # Extract v and w
    v_sol = solution[:, 0]
    w_sol = solution[:, 1]

    #Print Errors
    v_err=(v_sol-v)**2
    w_err=(w_sol-w)**2

    tot_err=(v_err+w_err)
    return tot_err

#NN for Parameter Estimation

#Set up Callbacks
bestmodel = K.callbacks.ModelCheckpoint(
    "C:\Anaconda3\FHN_RNN_Parameters.keras",
    monitor = "loss",
    verbose = 0,
    save_best_only = True,
    save_freq="epoch")

#Setup dataset
train_dataset = K.utils.timeseries_dataset_from_array(
    again[:20],
    targets=np.ones((20,3)),
    sequence_length=1,
    shuffle=True,
    batch_size=1)

inputs = K.Input(shape=(1, 2))
outputs = layers.GRU(3, activation='relu')(inputs)


modele = K.Model(inputs,outputs)

K.optimizers.RMSprop(
    learning_rate=0.000051,
    name="RMSprop")



modele.compile(
    optimizer="RMSprop"
    ,loss=my_loss_fn 
    ,run_eagerly=True
    )

history = modele.fit(train_dataset, 
           epochs=2,
           verbose=2,
           callbacks=[ bestmodel]
           )

#Make Parameter Predictions
modele.load_weights("C:\Anaconda3\FHN_RNN_Parameters.keras")
par = modele.predict(train_dataset)

a=par[0,0]
b=par[0,1]
c=par[0,2]      

print("a= ", a)
print("b= ", b)
print("c= ", c)

# System of ODEs
def system(y, t):
    v, w = y
    dvdt = (1/c)*(v-(1/3)*v**3-w)
    dwdt = c * (v - a*w + b)
    return [dvdt, dwdt]

# Initial conditions
v0 = -1
w0 = 1

# Solve ODE
solution = odeint(system, [v0, w0], t)

# Extract v and w
v_sol = solution[:, 0]
w_sol = solution[:, 1]

# Store results in DataFrame
data = pd.DataFrame({
    't': t,
    'v': v_sol,
    'w': w_sol
})

#Print Errors
v_err=(tf.math.squared_difference(v_sol, v))
w_err=(tf.math.squared_difference(w_sol, w))

v_err=tf.reduce_sum(v_err)
w_err=tf.reduce_sum(w_err)

tot_err=(v_err+w_err)/(2*len(t))
print(tot_err)

#Plotting Estimates
plt.figure()
plt.plot(t, v, "bo", label="v given")
plt.plot(t, w, "go", label="w given")
plt.plot(t,v_sol,"r-",label="Generated v")
plt.plot(t,w_sol,"r-",label="Generated w")
plt.legend()



dataerror = pd.DataFrame({
    't': t,
    'v-error': tf.math.subtract(v_sol, v),
    'w-error': tf.math.subtract(w_sol, w)
})
print(dataerror)