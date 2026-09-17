import numpy as np
import matplotlib.pyplot as plt

NX = 5
NY = 5
N = NX * NY

W_EXC = 0.5
PLASTICITY_AMOUNT = 0.1

I_EXT = 3.0
DECAY = 0.5

NT = 10000
DT = 0.1

TAU = 20.0
E_REST = -65.0
THETA = -55.0
R_M = 16.0

REFRACTORY_PERIOD = 2.0

plus_pattern = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])


def create_connections():

    connections = np.zeros((N, N), dtype=int)
    weights = np.zeros((N, N), dtype=float)

    flat_pattern = plus_pattern.flatten()

    for post in range(N):
        for pre in range(N):

            if post != pre and flat_pattern[post] == 1 and flat_pattern[pre] == 1:

                connections[post, pre] = 1
                weights[post, pre] = W_EXC

    return connections, weights


def set_input():

    external_input = np.zeros(N)

    flat_pattern = plus_pattern.flatten()

    for i in range(N):

        if flat_pattern[i] == 1:
            external_input[i] = I_EXT

    external_input[2 * NX + 0] = I_EXT * 1.2
    external_input[2 * NX + 1] = I_EXT * 1.2

    return external_input


def run_network(seed=23):

    connections, weights = create_connections()

    external_input = set_input()

    voltage = np.full(N, E_REST)

    synaptic_conductance = np.zeros(N)

    spike_state = np.zeros(N, dtype=int)

    spike_count = np.zeros(N, dtype=int)

    refractory_timer = np.zeros(N)

    rng = np.random.default_rng(seed)

    for t in range(NT):

        new_spike_state = np.zeros(N, dtype=int)

        refractory_timer = np.maximum(
            refractory_timer - DT,
            0
        )

        for i in range(N):

            recurrent_input = np.sum(
                connections[i, :] *
                weights[i, :] *
                spike_state
            )

            synaptic_conductance[i] = (
                DECAY * synaptic_conductance[i]
                + recurrent_input
            )

        for i in range(N):

            if refractory_timer[i] > 0:
                voltage[i] = E_REST
                continue

            random_input = rng.random()

            total_input = (
                external_input[i]
                + synaptic_conductance[i]
                + random_input
            )

            voltage[i] += DT * (
                -(voltage[i] - E_REST)
                + R_M * total_input
            ) / TAU

            if voltage[i] >= THETA:

                voltage[i] = E_REST

                new_spike_state[i] = 1

                spike_count[i] += 1

                refractory_timer[i] = REFRACTORY_PERIOD

        for pre in range(N):

            if new_spike_state[pre] == 1:

                for post in range(N):

                    if connections[post, pre] == 1:

                        weights[post, pre] += PLASTICITY_AMOUNT

        spike_state = new_spike_state

    return spike_count.reshape(NY, NX), weights


plus_result, final_weights = run_network()

print("PLUS PATTERN SPIKE COUNTS")
print(plus_result)

print("\nFINAL SYNAPTIC WEIGHTS")
print(final_weights)

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

plt.title("PLUS (+) Pattern with LIF and Synaptic Plasticity")

plt.tight_layout()
plt.show()