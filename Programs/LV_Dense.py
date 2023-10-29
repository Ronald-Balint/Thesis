# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 09:35:46 2023

@author: alexb
"""

from tensorflow import keras as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import square
import tensorflow.math as tfmath
from matplotlib import pyplot as plt
import numpy as np
import csv
from findiff import FinDiff
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
import pandas as pd
import math

#Master Reset
K.backend.clear_session()


# Open the file in 'r' mode, not 'rb'
csv_file = open('synthetic_data_LK.csv','r')
index = []
t = []
z1 =[]
z2=[]


# Read off and discard first line, to skip headers
csv_file.readline()

# Split columns while reading
for a, b,c,d in csv.reader(csv_file, delimiter=','):
    # Append each variable to a separate list
    index.append(float(a))
    t.append(float(b))
    z1.append(float(c))
    z2.append(float(d))


#Arranging data for NN to approximate curve    
data=t
train=np.column_stack((index,t))
train_data=train.tolist()
    
this_wont_work=np.column_stack((z1,z2))
again = this_wont_work.tolist()

#NN to Aproximate Curves
model0 = K.Sequential([
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(2, dtype='float32')
    ])

model0.compile(
    optimizer="adam",
    loss="mse"
    )

bestgraphmodel = K.callbacks.ModelCheckpoint(
    "C:\Anaconda3\LV_RNN_Curve.keras",
    monitor = "loss",
    verbose = 0,
    save_best_only = True,
    save_freq="epoch")

model0.fit(train_data, 
           again, 
           epochs=2,
           verbose=2
           #,callbacks=bestgraphmodel
           ,batch_size=len(t)
          )

#Make Parameter Predictions
model0.load_weights("C:\Anaconda3\LV_RNN_Curve.keras")

#Prediction of Curve
predict0=model0.predict(train_data)

#Graph various approximations of each method
plt.figure()
plt.plot(data, z1, "bo", label="z1")
plt.plot(data, z2, "go", label="z2")
plt.plot(data,predict0[:,0],"g*", label="z1 NN Prediction")
plt.plot(data,predict0[:,1],"r*", label="z2 NN Prediction")
plt.legend()
plt.show()

#Predictions
z1p=tf.cast(predict0[:,0].tolist(),tf.float32)
z2p=tf.cast(predict0[:,1].tolist(),tf.float32)


#Calculating Numerical Derivatives    
delta = t[1]-t[0]

d_dt = FinDiff(0,delta,dtype="float32")
dz1 = tf.cast(d_dt(predict0[:,0]).tolist(),tf.float32)
dz2 = tf.cast(d_dt(predict0[:,1]).tolist(),tf.float32)

#NN for Parameter Estimation
#Master Reset
K.backend.clear_session()

    
#Custom Loss Function  
def my_loss_fn(y_true, y_pred):

    a=float((y_pred[0,0]))
    b=float((y_pred[0,1]))
    c=float((y_pred[0,2]))
    d=float((y_pred[0,3]))  
     
    one = z1*(a-b*z2)
    two = z2*(c*z1-d)
      
    a1=(tfmath.subtract(dz1,one))
    a1=tfmath.abs(a1)
    b1=(tfmath.subtract(dz2,two))
    b1=tfmath.abs(b1)
    
    error = (tfmath.reduce_sum(a1)+tfmath.reduce_sum(b1))
    return error


#Setup dataset
superset = np.column_stack((train,again))
train_dataset = K.utils.timeseries_dataset_from_array(
    superset
    ,targets=np.ones((len(t),1))
    ,sequence_length=1
    ,sampling_rate=1
    ,batch_size=len(t)
    #,start_index=2
    #,end_index=5
    )

initializer = K.initializers.RandomUniform(minval=.4, maxval=.6, seed=20)

opt = K.optimizers.RMSprop(learning_rate=0.01)
#opt = K.mixed_precision.LossScaleOptimizer(opt)

modele = K.Sequential([
    Dense(4,activation="relu", dtype='float64')
    ])

modele.compile(
    optimizer=opt,
    loss=my_loss_fn
    ,run_eagerly=True
    )

history = modele.fit(
           again,
           np.ones([len(t),4]).tolist(),
           epochs=10000,
           verbose=2,
           batch_size=len(t)
           )

#Generate Parameter Estimates from last observation
par = modele.predict(again)



# Parameters
a=par[0,0]
b=par[0,1]
c=par[0,2]
d=par[0,3]
print("a= ", a)
print("b= ", b)
print("c= ", c)
print("d= ", d)


# System of ODEs
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
z1_s = solution[:, 0]
z2_s = solution[:, 1]

#Plotting Estimates
plt.figure()
plt.plot(t, z1, "bo", label="z1 given")
plt.plot(t, z2, "go", label="z2 given")
plt.plot(t,z1_s,"r-",label="Generated z1")
plt.plot(t,z2_s,"b-",label="Generated z2")
plt.plot(t,predict0[:,0],"k-", label="z1 NN Regression")
plt.plot(t,predict0[:,1],"k-", label="z2 NN Regression")
plt.legend()

#Print Errors
z1_e_err=(tf.math.squared_difference(z1_s, z1))
z2_e_err=(tf.math.squared_difference(z2_s, z2))

#Plotting Estimates
plt.figure()
plt.plot(t,z1_e_err,"r-",label="z1 Squared Difference")
plt.plot(t,z2_e_err,"g-",label="z2 Squared Difference")
plt.legend()

z1_err=tf.reduce_sum(z1_e_err)
z2_err=tf.reduce_sum(z2_e_err)

tot_err=(z1_err+z2_err)/(2*len(t))
print(tot_err)

dataerror = pd.DataFrame({
    't': t,
    'z1-error': tf.math.subtract(z1_s, z1),
    'z2-error': tf.math.subtract(z2_s, z2),
})
print(dataerror)