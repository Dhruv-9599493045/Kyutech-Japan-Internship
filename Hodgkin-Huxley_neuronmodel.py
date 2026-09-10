import math
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------

dt = 0.025       # time step [ms]
T = 100.0        # total simulation time [ms]

# Initial values
# v[0] = membrane voltage
# v[1] = m
# v[2] = h
# v[3] = n

v0 = np.array([-65.0, 0.05, 0.6, 0.32])


# -----------------------------
# Model parameters
# -----------------------------

cm = 1.0

gNa = 120.0
gK = 36.0
gL = 0.3

ena = 50.0
ek = -77.0
el = -54.387

Ibias = 10.0


# -----------------------------
# Ion channel functions
# -----------------------------

def alpha_m(v):
    return 0.1 * (v + 40.0) / (1.0 - math.exp(-(v + 40.0) / 10.0))


def beta_m(v):
    return 4.0 * math.exp(-(v + 65.0) / 18.0)


def alpha_h(v):
    return 0.07 * math.exp(-(v + 65.0) / 20.0)


def beta_h(v):
    return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))


def alpha_n(v):
    return 0.01 * (v + 55.0) / (1.0 - math.exp(-(v + 55.0) / 10.0))


def beta_n(v):
    return 0.125 * math.exp(-(v + 65.0) / 80.0)


# -----------------------------
# Ion currents
# -----------------------------

def I_Na(v):
    return gNa * v[1]**3 * v[2] * (v[0] - ena)


def I_K(v):
    return gK * v[3]**4 * (v[0] - ek)


def I_L(v):
    return gL * (v[0] - el)


# -----------------------------
# Differential equations
# -----------------------------

def dALLdt(v):

    dvdt = (Ibias - I_Na(v) - I_K(v) - I_L(v)) / cm

    dmdt = (
        alpha_m(v[0]) * (1.0 - v[1])
        - beta_m(v[0]) * v[1]
    )

    dhdt = (
        alpha_h(v[0]) * (1.0 - v[2])
        - beta_h(v[0]) * v[2]
    )

    dndt = (
        alpha_n(v[0]) * (1.0 - v[3])
        - beta_n(v[0]) * v[3]
    )

    return np.array([dvdt, dmdt, dhdt, dndt])


# -----------------------------
# Number of simulation steps
# -----------------------------

nt = int(T / dt + 0.5)


# -----------------------------
# Arrays to store results
# -----------------------------

t_values = np.zeros(nt + 1)

v_values = np.zeros(nt + 1)
m_values = np.zeros(nt + 1)
h_values = np.zeros(nt + 1)
n_values = np.zeros(nt + 1)


# Initial conditions

t_values[0] = 0.0

v_values[0] = v0[0]
m_values[0] = v0[1]
h_values[0] = v0[2]
n_values[0] = v0[3]


# -----------------------------
# RK4 METHOD
# -----------------------------

for i in range(nt):

    # Current state
    v = np.array([
        v_values[i],
        m_values[i],
        h_values[i],
        n_values[i]
    ])


    # k1
    k1 = dALLdt(v)


    # k2
    k2 = dALLdt(v + 0.5 * k1 * dt)


    # k3
    k3 = dALLdt(v + 0.5 * k2 * dt)


    # k4
    k4 = dALLdt(v + k3 * dt)


    # RK4 final update
    v_new = v + dt * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    ) / 6.0


    # Store results
    t_values[i + 1] = t_values[i] + dt

    v_values[i + 1] = v_new[0]
    m_values[i + 1] = v_new[1]
    h_values[i + 1] = v_new[2]
    n_values[i + 1] = v_new[3]


# -----------------------------
# Print results
# -----------------------------

print("     t          v          m          h          n")

for i in range(nt + 1):
    print(
        f"{t_values[i]:8.2f} "
        f"{v_values[i]:8.5f} "
        f"{m_values[i]:8.5f} "
        f"{h_values[i]:8.5f} "
        f"{n_values[i]:8.5f}"
    )


# -----------------------------
# Voltage vs Time
# -----------------------------

plt.figure()

plt.plot(t_values, v_values)

plt.xlabel("t [ms]")
plt.ylabel("v [mV]")
plt.title("Hodgkin-Huxley Neuron")

plt.grid(True)

plt.show()


# -----------------------------
# Phase Plane
# -----------------------------

plt.figure()

plt.plot(v_values, m_values)

plt.xlabel("v [mV]")
plt.ylabel("m")

plt.title("Hodgkin-Huxley Phase Plane")

plt.grid(True)

plt.show()