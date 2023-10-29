# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 09:35:46 2023

@author: alexb
"""

from tensorflow import keras as K
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from findiff import FinDiff
import numpy as np
from scipy.interpolate import CubicSpline
import csv
import sys

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


#Cubic Spline Estimation of Curve
cs1 = CubicSpline(t,v)
cs2 = CubicSpline(t,w)
cs1half = cs1(t)
cs2half = cs2(t)


#Plotting Estimates
plt.figure()
plt.plot(data, v, "bo", label="v")
plt.plot(data, w, "go", label="w")

plt.plot(t,cs1half, "r--", label="v Spline Prediction")
plt.plot(t,cs2half, "b--", label="w Spline Prediction")
plt.legend()


#Calculating Numerical Derivatives

delta = t[1]-t[0]

d_dt = FinDiff(0,delta,1, acc=4)
dv_dt = d_dt(cs1half)
dw_dt = d_dt(cs2half)

#Plotting Estimates
plt.figure()
plt.plot(t,dv_dt, "r--", label="dv Prediction")
plt.plot(t,dw_dt, "b--", label="dw Prediction")

plt.legend()

#Changing data types for NN processing
cs1real=[]
cs2real=[]
for real in range(0,len(t)):
    half1=float(v[real])
    half2=float(w[real])
    cs1real.append(half1)
    cs2real.append(half2)
    
#Setting up NN inputs and Outputs for Parameter Estimation    

#Defining Custom Loss Function  
def my_loss_fn(y_true, y_pred):
    triallist =[]
    a=y_pred[0,0]
    b=y_pred[0,1]
    c=y_pred[0,2]
    for multiply in range (0,len(t)):
        eqn1 = dv_dt[multiply]-(c*(v[multiply]-(1/3)*(v[multiply]**3)+w[multiply]))
        eqn2 = dw_dt[multiply]+(1/c)*(v[multiply]-a-(b*w[multiply]))
        one = float(eqn1)
        two = float(eqn2)
        trial = (one**2+two**2)
        triallist.append(trial)
    sumtrial = sum(triallist)#/len(triallist)
    return sumtrial 

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
    targets=np.zeros((len(t),1)),
    sequence_length=20,
    shuffle=True,
    batch_size=1)

inputs = K.Input(shape=(20, 2))
x = layers.GRU(80, activation='tanh', return_sequences=True)(inputs)
x = layers.GRU(80, activation='tanh', return_sequences=True)(x)
x = layers.GRU(80, activation='tanh', return_sequences=True)(x)
x = layers.GRU(80, activation='tanh', return_sequences=True)(x)
x = layers.GRU(80, activation='tanh', return_sequences=True)(x)
x = layers.GRU(80, activation='tanh', return_sequences=True)(x)
x = layers.GRU(80, activation='tanh', return_sequences=True)(x)
outputs = layers.GRU(3, activation='linear')(x)


modele = K.Model(inputs,outputs)

modele.compile(
    optimizer="adam",
    loss=my_loss_fn,
    )

history = modele.fit(train_dataset, 
           epochs=10000,
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
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.title("Training loss")
plt.legend()
plt.show()


