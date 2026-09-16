import numpy as np
import matplotlib.pyplot as plt

NX = 5
NY = 5
N = NX * NY

TAU = 20.0
W_EXC = 5.0
E_REST = -65.0
THETA = -55.0
R_M = 16.0
I_EXT = 1.5
DECAY = 0.5

NT = 1000
DT = 1.0

cross_pattern = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])

def create_connections():
    weights = np.zeros((N, N))
    flat_pattern = cross_pattern.flatten()

    for post in range(N):
        for pre in range(N):
            if post != pre:
                weights[post, pre] += (
                    W_EXC *
                    flat_pattern[post] *
                    flat_pattern[pre]
                )

    return weights

def set_input():
    external_input = np.zeros(N)

    index = 4 * NX + 0
    external_input[index] = I_EXT

    return external_input

def run_network(seed=23):
    weights = create_connections()
    external_input = set_input()

    voltage = np.full(N, E_REST)
    synaptic_conductance = np.zeros(N)
    spike_state = np.zeros(N, dtype=int)
    spike_count = np.zeros(N, dtype=int)

    rng = np.random.default_rng(seed)

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

            voltage[i] += DT * (
                -(voltage[i] - E_REST)
                + synaptic_conductance[i]
                + R_M * (
                    external_input[i]
                    + random_input
                )
            ) / TAU

            if voltage[i] > THETA:
                new_spike_state[i] = 1
                spike_count[i] += 1
                voltage[i] = E_REST

        spike_state = new_spike_state

    return spike_count.reshape(NY, NX)

cross_result = run_network()

print("CROSS PATTERN SPIKE COUNTS")
print(cross_result)

plt.figure(figsize=(7, 7))

plt.imshow(
    cross_result,
    cmap="viridis",
    interpolation="nearest"
)

plt.colorbar(label="Spike Count")

for i in range(NY):
    for j in range(NX):
        plt.text(
            j,
            i,
            str(cross_result[i, j]),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

plt.xticks(range(NX), range(NX))
plt.yticks(range(NY), range(NY))

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Hopfield Network - CROSS")

plt.tight_layout()
plt.show()