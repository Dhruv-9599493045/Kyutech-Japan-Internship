import numpy as np
import matplotlib.pyplot as plt 

def f(S,tau,input_signal):
    return -S / tau + input_signal

T=100
h=0.1
tau=10

n = int (T/h)

t_values = np.zeros(n+1)
S2_values= np.zeros(n+1)
S1_values= np.zeros(n+1)

S_total_values= np.zeros(n+1)
input1_values = np.zeros(n+1)
input2_values = np.zeros(n+1)
input_total_values = np.zeros(n+1)

t_values[0] = 0


for k in range(n):
    t = t_values[k]

    if 10<= t <= 15:
        input1=1
    else:
        input1 = 0

    if 30 <= t <=35:
        input2 = 1
    else:
        input2=0


    input1_values[k] = input1
    input2_values[k] = input2


    ds1 = f(S1_values[k],tau,input1)
    ds2 = f(S2_values[k],tau,input2)

    t_values[k+1]=t_values[k] + h

    S1_values[k+1]= S1_values[k] + h * ds1
    S2_values[k+1]= S2_values[k] + h * ds2

    S_total_values[k+1]=(
        S1_values[k+1] + S2_values[k+1]
    )

    input_total_values[k]=input1 + input2

input_total_values[n]=input1_values[n]+input2_values[n]

print("    T    S1     s2     S_Total")

for i in range(n+1):
    print(
        f"{t_values[i]:8.2f}"
        f"{S1_values[i]:12.6f}"
        f"{S2_values[i]:12.6f}"
        f"{S_total_values[i]:12.6f}"
    )

plt.figure()
plt.plot(t_values,S1_values)
plt.xlabel("t")
plt.ylabel("S1")
plt.title("Response to first input pulse")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_values,S2_values)
plt.xlabel("T")
plt.ylabel("S2")
plt.title("Response to second Input Pulse")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_values,S_total_values)
plt.xlabel("T")
plt.ylabel("S1+S2")
plt.title("Sumation of Two Synapic Response")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_values,S2_values,label = "S1")
plt.plot(t_values,S2_values, label = "S2")
plt.plot(t_values,S_total_values,label = "S1+S2")
plt.xlabel("T")
plt.ylabel("Synaptic Signal")
plt.title("Two input pulses and ther summation")

plt.legend()
plt.grid(True)
plt.show()
