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
data=t
train=np.column_stack((index,t))
train_data=train.tolist()

#Grouping columns and changing to list form
this_wont_work=np.column_stack((v,w))
again = this_wont_work.tolist()

#Structure of Network
model0 = K.Sequential([
    layers.Dense(80, activation="tanh", dtype='float64'),
    layers.Dense(80, activation="tanh", dtype='float64'),
    layers.Dense(80, activation="tanh", dtype='float64'),
    layers.Dense(80, activation="tanh", dtype='float64'),
    layers.Dense(80, activation="tanh", dtype='float64'),
    layers.Dense(80, activation="tanh", dtype='float64'),
    layers.Dense(2, activation='linear',dtype='float64')
    ])

bestgraphmodel = K.callbacks.ModelCheckpoint(
    "C:\Anaconda3\FHN_RNN_Graph.keras",
    monitor = "loss",
    verbose = 0,
    save_best_only = True,
    save_freq="epoch")

model0.compile(
    optimizer="adam",
    loss="mse"
    )

model0.fit(train_data, 
           again, 
           epochs=5000,
           verbose=2,
           callbacks=bestgraphmodel
          )
model0.load_weights("C:\Anaconda3\FHN_RNN_Graph.keras")
predict0=model0.predict(train_data)

#Plotting Estimates
plt.figure()
plt.plot(data, v, "bo", label="v")
plt.plot(data, w, "go", label="w")
plt.plot(data,predict0[:,0],"green", label="v NN Prediction")
plt.plot(data,predict0[:,1],"red", label="w NN Prediction")
plt.legend()


#Calculating Numerical Derivatives

delta = t[1]-t[0]

d_dt = FinDiff(0,delta,1)
dv_dt = tf.cast(d_dt(predict0[:,0]),tf.float32)
dw_dt = tf.cast(d_dt(predict0[:,1]),tf.float32)

#Plotting Estimates
plt.figure()
plt.plot(t,dv_dt, "r--", label="dv Prediction")
plt.plot(t,dw_dt, "b--", label="dw Prediction")

plt.legend()

   
#Setting up NN inputs and Outputs for Parameter Estimation    

#Defining Custom Loss Function  
def my_loss_fn(y_true, y_pred):
    a=y_pred[0,0]
    b=y_pred[0,1]
    c=y_pred[0,2]
    cube=np.full(len(v),3)
    ick_1rhs = math.multiply(math.reciprocal(c),math.subtract(math.subtract(v,math.divide(math.pow(v,cube),3)),w))
    ick_1 = math.subtract(dv_dt,ick_1rhs)
    ick_2rhs = math.multiply(c,(math.add(math.subtract(v,math.multiply(a,w)),b)))
    ick_2 = math.subtract(dw_dt,ick_2rhs)
    squ1 = math.square(ick_1)
    squ2 = math.square(ick_2)
    finalick = math.add(squ1,squ2)
    return finalick

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
    this_wont_work[:20],
    targets=np.ones((1,3)),
    sequence_length=20,
    shuffle=True,
    batch_size=1)

inputs = K.Input(shape=(20, 2))
x = layers.LSTM(4, activation='tanh', return_sequences=True)(inputs)
x = layers.LSTM(4, activation='tanh', return_sequences=True)(x)
x = layers.LSTM(4, activation='tanh', return_sequences=True)(x)
x = layers.LSTM(4, activation='tanh', return_sequences=True)(x)
x = layers.LSTM(4, activation='tanh', return_sequences=True)(x)
x = layers.LSTM(4, activation='tanh', return_sequences=True)(x)
x = layers.LSTM(4, activation='tanh', return_sequences=True)(x)
x = layers.LSTM(4, activation='tanh', return_sequences=True)(x)
outputs = layers.LSTM(3, activation='linear')(x)


modele = K.Model(inputs,outputs)

K.optimizers.RMSprop(
    learning_rate=0.000051,
    name="RMSprop")



modele.compile(
    optimizer="RMSprop",
    loss=my_loss_fn
    )

history = modele.fit(train_dataset, 
           epochs=50000,
           verbose=2,
           callbacks=[ bestmodel]
           )

#Make Parameter Predictions
modele.load_weights("C:\Anaconda3\FHN_RNN_Parameters.keras")
par = modele.predict(train_dataset)

triallist =[]
a=par[0,0]
b=par[0,1]
c=par[0,2]      

print("a= ", a)
print("b= ", b)
print("c= ", c)


loss = history.history["loss"]
epochs = range(1,len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "o", color="orange")
plt.title("2nd Half Training loss")
plt.show()

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

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

#Plotting Estimates
plt.figure()
plt.plot(t, v, "bo", label="v given")
plt.plot(t, w, "go", label="w given")
plt.plot(t,v_sol,"r-",label="Generated v")
plt.plot(t,w_sol,"r-",label="Generated w")
plt.legend()

#Print Errors
v_err=(tf.math.squared_difference(v_sol, v))
w_err=(tf.math.squared_difference(w_sol, w))

v_err=tf.reduce_sum(v_err)
w_err=tf.reduce_sum(w_err)

tot_err=(v_err+w_err)/len(t)
print(tot_err)

dataerror = pd.DataFrame({
    't': t,
    'v-error': tf.math.subtract(v_sol, v),
    'w-error': tf.math.subtract(w_sol, w)
})
print(dataerror)