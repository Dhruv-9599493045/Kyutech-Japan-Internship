import numpy as np
import matplotlib.pyplot as plt

DT = 0.1
T = 1000.0

A_PLUS = 0.01
A_MINUS = 0.0105

TAU_PLUS = 20.0
TAU_MINUS = 20.0

W_INITIAL = 0.5

SPIKE_PROBABILITY = 0.01

time = np.arange(0, T, DT)

pre_trace = np.zeros(len(time))
post_trace = np.zeros(len(time))

weight = W_INITIAL

rng = np.random.default_rng(23)

for t in range(1, len(time)):

    pre_spike = rng.random() < SPIKE_PROBABILITY
    post_spike = rng.random() < SPIKE_PROBABILITY

    pre_trace[t] = (
        pre_trace[t - 1] *
        np.exp(-DT / TAU_PLUS)
    )

    post_trace[t] = (
        post_trace[t - 1] *
        np.exp(-DT / TAU_MINUS)
    )

    if post_spike:
        weight += A_PLUS * pre_trace[t]

    if pre_spike:
        weight -= A_MINUS * post_trace[t]

    if pre_spike:
        pre_trace[t] += 1

    if post_spike:
        post_trace[t] += 1

    if weight < 0:
        weight = 0


delta_times = np.arange(-100, 101, 1)

delta_weights = np.zeros(len(delta_times))

for i, delta_t in enumerate(delta_times):

    if delta_t > 0:

        delta_weights[i] = (
            A_PLUS *
            np.exp(-delta_t / TAU_PLUS)
        )

    elif delta_t < 0:

        delta_weights[i] = (
            -A_MINUS *
            np.exp(delta_t / TAU_MINUS)
        )

    else:

        delta_weights[i] = 0


plt.figure(figsize=(12, 5))

plt.plot(
    time,
    pre_trace,
    label="Presynaptic trace"
)

plt.plot(
    time,
    post_trace,
    label="Postsynaptic trace"
)

plt.xlabel("Time (ms)")
plt.ylabel("Activity trace")
plt.title("Online STDP Activity Traces")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    delta_times,
    delta_weights,
    linewidth=2
)

plt.axhline(
    0,
    linestyle="--"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Δt (ms)")
plt.ylabel("Δw")
plt.title("STDP Learning Window")

plt.grid(True)

plt.tight_layout()
plt.show()

print("Initial synaptic weight:", W_INITIAL)
print("Final synaptic weight:", weight)