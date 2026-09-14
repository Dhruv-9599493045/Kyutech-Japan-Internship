import numpy as np
import matplotlib.pyplot as plt

tau_s = 20
dt=0.05
T=100

n = int(T/dt)

t_values = np.zeros(n+1)
r_values = np.zeros(n+1)

t_values[0]=0
r_values[0]=0

for k in range(n):
    if k==0:
        spike = 1
    else:
        spike = 0

    r_values[k+1] = (
        (1-dt/tau_s)*r_values[k] + spike/tau_s
    )

    t_values[k+1]=t_values[k] + dt

plt.figure(figsize=(8,5))
plt.plot(t_values,r_values)

plt.xlabel("Time(ms)")
plt.ylabel("Synaptic Response r(t)")
plt.title("Single Exponential Synapse")
plt.grid(True)
plt.show()
