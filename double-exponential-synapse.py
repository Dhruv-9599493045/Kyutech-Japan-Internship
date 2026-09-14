import numpy as np
import matplotlib.pyplot as plt 

tau_r = 2
tau_d = 20
dt=0.05
T=100

n= int(T/dt) 

t_values = np.zeros(n+1)
r_values = np.zeros(n+1)
h_values = np.zeros(n+1)

t_values[0]=0
r_values[0]=0
h_values[0]=0

for k in range(n):
    if k == 0:
        spike = 1
    else:
        spike=0

    r_new=(
        (1-dt/tau_d)*r_values[k]+h_values[k]*dt
    )
    h_new=(
        (1-dt/tau_r)*h_values[k]+spike/(tau_r*tau_d)
    )

    r_values[k+1]=r_new
    h_values[k+1]=h_new

    t_values[k+1] = t_values[k]+dt

plt.figure(figsize=(8,5))
plt.plot(t_values,r_values)

plt.xlabel("Time(ms)")
plt.ylabel("Synaptic Response r(t)")
plt.title("Double Exponential Synapse")
plt.grid(True)
plt.show()

