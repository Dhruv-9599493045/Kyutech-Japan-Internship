import numpy as np
import matplotlib.pyplot as plt 

def f(S,tau,input_signal):
    return -S / tau + input_signal

T=100
h=0.1
tau=10

n = int (T/h)

t_values = np.zeros(n+1)
S_values= np.zeros(n+1)
input_values = np.zeros(n+1)

t_values[0] = 0
S_values[0] = 0

for k in range(n):
    t = t_values[k]

    if 10<= t <= 15:
        input_signal=1
    else:
        input_signal = 0

    input_values[k] = input_signal

    ds = f(S_values[k],tau,input_signal)

    t_values[k+1]=t_values[k] + h
    S_values[k+1]= S_values[k] + h * ds

input_values[n]=0

print("    T    S     INPUT")

for i in range(n+1):
    print(
        f"{t_values[i]:8.2f}"
        f"{S_values[i]:12.6f}"
        f"{input_values[i]:12.2f}"
    )

plt.figure()
plt.plot(t_values,S_values)
plt.xlabel("t")
plt.ylabel("S")
plt.title("Synaptic Signal S(t)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_values,input_values)
plt.xlabel("T")
plt.ylabel("Input")
plt.title("Input Pulse")
plt.grid(True)
plt.show()
