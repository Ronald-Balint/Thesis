# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 09:35:46 2023

@author: alexb
"""

from tensorflow import keras as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.math import * 
from matplotlib import pyplot as plt
import numpy as np
import csv
from findiff import FinDiff
from scipy.integrate import odeint
import pandas as pd

#Master Reset
K.backend.clear_session()


# Open the file in 'r' mode, not 'rb'
csv_file = open('synthetic_data_Bellman.csv','r')
index = []
t = []
z =[]


# Read off and discard first line, to skip headers
csv_file.readline()

# Split columns while reading
for a, b,c in csv.reader(csv_file, delimiter=','):
    # Append each variable to a separate list
    index.append(float(a))
    t.append(float(b))
    z.append(float(c))


#Arranging data for NN to approximate curve    
data=t
train=np.column_stack((index,t))
train_data=train.tolist()
    

again = z

#NN to Aproximate Curves
model0 = K.Sequential([
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(80, activation="sigmoid", dtype='float32'),
    Dense(1, dtype='float32')
    ])

model0.compile(
    optimizer="adam",
    loss="mse"
    )

bestgraphmodel = K.callbacks.ModelCheckpoint(
    "C:\Anaconda3\Bellman_RNN_Curve.keras",
    monitor = "loss",
    verbose = 0,
    save_best_only = True,
    save_freq="epoch")

model0.fit(train_data, 
           again, 
           epochs=1,
           verbose=0
           #,callbacks=bestgraphmodel
           ,batch_size=len(t)
          )

#Make Parameter Predictions
model0.load_weights("C:\Anaconda3\Bellman_RNN_Curve.keras")

#Prediction of Curve
predict0=model0.predict(train_data)

#Graph various approximations of each method
plt.figure()
plt.plot(data, z, "bo", label="z")
plt.plot(data,predict0[:,0],"g*", label="z NN Prediction")
plt.legend()
plt.show()

#Predictions
zp=tf.cast(predict0[:,0].tolist(),tf.float32)



#Calculating Numerical Derivatives    
delta = t[1]-t[0]

d_dt = FinDiff(0,delta,dtype="float32")
dz = tf.cast(d_dt(predict0[:,0]).tolist(),tf.float32)


#NN for Parameter Estimation
#Master Reset
K.backend.clear_session()

    
#Custom Loss Function  
def my_loss_fn(y_true, y_pred):

    k1=float((y_pred[0,0]))
    k2=float((y_pred[0,1]))
 
     
    one = subtract(scalar_mul(k1,multiply(subtract(126.2,zp),square(subtract(91.9,zp)))),(scalar_mul(k2,square(zp))))
          
    a1=(subtract(dz,one))
    a1=square(a1)
    
    error = reduce_sum(a1)
    return error


#Setup dataset
superset = np.column_stack((train,again))
train_dataset = K.utils.timeseries_dataset_from_array(
    superset
    ,targets=np.ones((len(t),2))
    ,sequence_length=1
    ,sampling_rate=1
    ,batch_size=int(len(t)/1)
    )


opt = K.optimizers.RMSprop(learning_rate=0.0000001)
opt = K.mixed_precision.LossScaleOptimizer(opt)

inputs = K.Input(shape=(1, 3))
x=LSTM(3, activation='tanh', dtype="float32",return_sequences=True)(inputs)
x=LSTM(3, activation='tanh', dtype="float32",return_sequences=True)(x)
x=LSTM(3, activation='tanh', dtype="float32",return_sequences=True)(x)
x=LSTM(3, activation='tanh', dtype="float32",return_sequences=True)(x)
x=LSTM(3, activation='tanh', dtype="float32",return_sequences=True)(x)
outputs = LSTM(2, activation='tanh', dtype="float32")(x)


modele = K.Model(inputs,outputs)

modele.compile(
    optimizer=opt,
    loss=my_loss_fn
    ,run_eagerly=True
    )
cb1 = K.callbacks.ModelCheckpoint(
    "C:\Anaconda3\Bellman_RNN_Params.keras",
    monitor = "loss",
    verbose = 0,
    save_best_only = True,
    save_freq="epoch")


history = modele.fit(train_dataset, 
           epochs=10000,
           verbose=2,
           callbacks=cb1
           )

#Generate Parameter Estimates from last observation
modele.load_weights("C:\Anaconda3\Bellman_RNN_Params.keras")
par = modele.predict(train_dataset)



# Parameters
k1=par[0,0]
k2=par[0,1]

print("k1= ", k1)
print("k2= ", k2)


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
zs = solution[:, 0]

#Plotting Estimates
plt.figure()
plt.plot(t, z, "bo", label="z given")
plt.plot(t,zs,"r-",label="Generated z1")
plt.legend()

#Print Errors
z_e_err=(tf.math.squared_difference(zs, z))

#Plotting Estimates
plt.figure()
plt.plot(t,z_e_err,"r-",label="z Squared Difference")

plt.legend()

z_err=tf.reduce_sum(z_e_err)


tot_err=(z_err)/(len(t))
print(tot_err)

dataerror = pd.DataFrame({
    't': t,
    'z-error': tf.math.subtract(zs, z),
})
print(dataerror)