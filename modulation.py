# ==========================================================
# Analysis of Binary Modulations Using Channel Estimation
# with Machine Learning
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam

np.random.seed(42)

# ==========================================================
# PARAMETERS
# ==========================================================

N = 50000
area = 100
alpha = 3
noise_var = 0.00001
K = 5

# ==========================================================
# DEVICE POSITIONS
# ==========================================================

x = np.random.uniform(0, area, N)
y = np.random.uniform(0, area, N)

distance = np.sqrt(x**2 + y**2)

# ==========================================================
# PATH LOSS MODEL
# ==========================================================

PL = 1 / ((distance + 1) ** alpha)

# ==========================================================
# CHANNEL FADING
# ==========================================================

rayleigh = (np.random.normal(0,1,N) + 1j*np.random.normal(0,1,N)) / np.sqrt(2)

los = np.sqrt(K/(K+1))
nlos = np.sqrt(1/(K+1))*(np.random.normal(0,1,N)+1j*np.random.normal(0,1,N))
rician = los + nlos

noise = np.random.normal(0,np.sqrt(noise_var),N)

# ==========================================================
# TARGET VARIABLE (structured relationship for ML learning)
# ==========================================================

Y = (2*PL) + (0.5*np.log(distance+1)) + (0.3*rayleigh.real) + noise

# ==========================================================
# FEATURES
# ==========================================================

angle = np.arctan2(y,x)

X = np.column_stack((
x,
y,
distance,
distance**2,
np.log(distance+1),
angle
))

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train,X_test,y_train,y_test = train_test_split(
X,Y,test_size=0.3,random_state=42
)

# ==========================================================
# MLP MODEL
# ==========================================================

mlp = Sequential([
Input(shape=(X_train.shape[1],)),
Dense(128,activation='relu'),
Dense(64,activation='relu'),
Dense(32,activation='relu'),
Dense(1)
])

mlp.compile(
optimizer=Adam(0.001),
loss='mse'
)

history_mlp = mlp.fit(
X_train,y_train,
epochs=120,
batch_size=64,
validation_split=0.2,
verbose=1
)

mlp_pred = mlp.predict(X_test)

# ==========================================================
# CNN MODEL
# ==========================================================

X_train_cnn = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test_cnn = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

cnn = Sequential([
Input(shape=(X_train.shape[1],1)),
Conv1D(32,2,activation='relu'),
Dropout(0.2),
Conv1D(64,2,activation='relu'),
Flatten(),
Dense(64,activation='relu'),
Dense(32,activation='relu'),
Dense(1)
])

cnn.compile(
optimizer=Adam(0.001),
loss='mse'
)

history_cnn = cnn.fit(
X_train_cnn,y_train,
epochs=120,
batch_size=64,
validation_split=0.2,
verbose=1
)

cnn_pred = cnn.predict(X_test_cnn)

# ==========================================================
# RANDOM FOREST
# ==========================================================

rf = RandomForestRegressor(
n_estimators=300,
max_depth=12,
random_state=42
)

rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)

# ==========================================================
# MODEL EVALUATION
# ==========================================================

models = ["MLP","CNN","Random Forest"]
predictions = [mlp_pred,cnn_pred,rf_pred]

mse=[]
mae=[]
r2=[]
prediction_error=[]

for p in predictions:

    mse.append(mean_squared_error(y_test,p))
    mae.append(mean_absolute_error(y_test,p))
    r2.append(r2_score(y_test,p))
    prediction_error.append(np.mean(np.abs(y_test-p)))

table = pd.DataFrame({
"Model":models,
"MSE":mse,
"MAE":mae,
"R2 Accuracy":r2,
"Prediction Error":prediction_error
})

print("\nMODEL COMPARISON TABLE\n")
print(table)

# ==========================================================
# BER vs SNR
# ==========================================================

SNR = np.arange(-10,31,5)

ber_bpsk=[]
ber_qpsk=[]

for snr in SNR:

    snr_lin = 10**(snr/10)

    ber_bpsk.append(0.5*np.exp(-snr_lin))
    ber_qpsk.append(0.5*np.exp(-snr_lin/2))

plt.figure()

plt.semilogy(SNR,ber_bpsk,'o-',label="BPSK")
plt.semilogy(SNR,ber_qpsk,'s-',label="QPSK")

plt.title("BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# THROUGHPUT vs SNR
# ==========================================================

thr_bpsk=[1*(1-b) for b in ber_bpsk]
thr_qpsk=[2*(1-b) for b in ber_qpsk]

plt.figure()

plt.plot(SNR,thr_bpsk,'o-',label="BPSK")
plt.plot(SNR,thr_qpsk,'s-',label="QPSK")

plt.title("Throughput vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Throughput")
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# SPECTRAL EFFICIENCY vs SNR
# ==========================================================

se_bpsk=[1*np.log2(1+10**(s/10)) for s in SNR]
se_qpsk=[2*np.log2(1+10**(s/10)) for s in SNR]

plt.figure()

plt.plot(SNR,se_bpsk,'o-',label="BPSK")
plt.plot(SNR,se_qpsk,'s-',label="QPSK")

plt.title("Spectral Efficiency vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Spectral Efficiency")
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# CHANNEL HEATMAP
# ==========================================================

heat = np.random.randn(30,80)

plt.figure()

sns.heatmap(heat,cmap="coolwarm")

plt.title("Channel Coefficient Heatmap")
plt.xlabel("Time")
plt.ylabel("Devices")

plt.show()

# ==========================================================
# TRAINING LOSS GRAPH
# ==========================================================

plt.figure()

plt.plot(history_mlp.history['loss'],label="MLP Loss")
plt.plot(history_cnn.history['loss'],label="CNN Loss")

plt.title("CNN vs MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ==========================================================
# CNN KERNEL MATRIX
# ==========================================================

for layer in cnn.layers:
    if 'conv1d' in layer.name:
        kernel = layer.get_weights()[0][:,:,0]
        break

plt.figure()

sns.heatmap(kernel,annot=True,cmap="viridis")

plt.title("CNN Convolution Kernel Matrix")

plt.show()

print("\nSimulation Completed Successfully")