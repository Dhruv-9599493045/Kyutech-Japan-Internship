import numpy as np
import matplotlib.pyplot as plt

NX = 5
NY = 5
N = NX * NY

W_EXC = 5.0
I_EXT = 10.0
DECAY = 0.5

NT = 10000
DT = 0.1

a = 0.1
b = 0.2
c = -65.0
d = 2.0

plus_pattern = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])


def create_connections():

    weights = np.zeros((N, N))
    flat_pattern = plus_pattern.flatten()

    for post in range(N):
        for pre in range(N):

            if post != pre:

                weights[post, pre] += (
                    W_EXC
                    * flat_pattern[post]
                    * flat_pattern[pre]
                )

    return weights


def set_input():

    external_input = np.zeros(N)
    flat_pattern = plus_pattern.flatten()

    for i in range(N):

        if flat_pattern[i] == 1:
            external_input[i] = I_EXT

    # Extra input to the left two neurons
    external_input[2 * NX + 0] = I_EXT * 1.5
    external_input[2 * NX + 1] = I_EXT * 1.5

    return external_input


def run_network(seed=23):

    weights = create_connections()
    external_input = set_input()

    voltage = np.full(N, c)
    recovery = np.full(N, b * c)

    synaptic_conductance = np.zeros(N)

    spike_state = np.zeros(N, dtype=int)
    spike_count = np.zeros(N, dtype=int)

    rng = np.random.default_rng(seed)

    voltage_history_1 = np.zeros(NT)
    voltage_history_2 = np.zeros(NT)

    for t in range(NT):

        new_spike_state = np.zeros(N, dtype=int)

        for i in range(N):

            recurrent_input = np.sum(
                weights[i, :] * spike_state
            )

            synaptic_conductance[i] = (
                DECAY * synaptic_conductance[i]
                + recurrent_input
            )

        for i in range(N):

            random_input = rng.random()

            total_input = (
                external_input[i]
                + synaptic_conductance[i]
                + random_input
            )

            dv = (
                0.04 * voltage[i] ** 2
                + 5 * voltage[i]
                + 140
                - recovery[i]
                + total_input
            )

            du = a * (
                b * voltage[i]
                - recovery[i]
            )

            previous_voltage = voltage[i]

            voltage[i] += DT * dv
            recovery[i] += DT * du

            if previous_voltage < 30 and voltage[i] >= 30:

                voltage[i] = c
                recovery[i] += d

                new_spike_state[i] = 1
                spike_count[i] += 1

        spike_state = new_spike_state

        voltage_history_1[t] = voltage[2 * NX + 0]
        voltage_history_2[t] = voltage[2 * NX + 1]

    return (
        spike_count.reshape(NY, NX),
        voltage_history_1,
        voltage_history_2
    )


# Run simulation
plus_result, voltage_1, voltage_2 = run_network()


# --------------------------------------------------
# SPIKE COUNT HEATMAP
# --------------------------------------------------

print("PLUS PATTERN SPIKE COUNTS")
print(plus_result)

plt.figure(figsize=(7, 7))

plt.imshow(
    plus_result,
    cmap="viridis",
    interpolation="nearest"
)

plt.colorbar(label="Spike Count")

for i in range(NY):
    for j in range(NX):

        plt.text(
            j,
            i,
            str(plus_result[i, j]),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

plt.xticks(range(NX), range(NX))
plt.yticks(range(NY), range(NY))

plt.xlabel("X position")
plt.ylabel("Y position")

plt.title("Izhikevich Fast-Spiking Model - PLUS (+) Pattern")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# MEMBRANE POTENTIAL OF TWO NEURONS
# --------------------------------------------------

time = np.arange(NT) * DT

# Show only first 100 ms for clarity
show_time = 100
samples = int(show_time / DT)

plt.figure(figsize=(12, 6))

plt.plot(
    time[:samples],
    voltage_1[:samples],
    label="Neuron (2,0)",
    linewidth=2
)

plt.plot(
    time[:samples],
    voltage_2[:samples],
    label="Neuron (2,1)",
    linewidth=2
)

plt.axhline(
    30,
    linestyle="--",
    linewidth=1.5,
    label="Spike threshold (30 mV)"
)

plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Membrane Potential (mV)", fontsize=12)

plt.title(
    "Membrane Potential of Two PLUS Neurons",
    fontsize=14
)

plt.xlim(0, show_time)
plt.ylim(-70, 35)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()