import numpy as np
import matplotlib.pyplot as plt

NX = 5
NY = 5
N = NX * NY

W_EXC = 0.5
PLASTICITY_AMOUNT = 0.1
WEIGHT_THRESHOLD = 5.0
I_EXT = 3.0
DECAY = 0.5

NT_LEARNING = 10000
NT_RECALL = 5000
DT = 0.1

TAU = 20.0
E_REST = -65.0
THETA = -55.0
R_M = 16.0
REFRACTORY_PERIOD = 2.0

PLUS_NEURONS = [2, 7, 10, 11, 12, 13, 14, 17, 22]
LEARNING_STIMULATION = PLUS_NEURONS
RECALL_STIMULATION = [10, 11]


def create_connections():

    connections = np.zeros((N, N), dtype=int)
    weights = np.zeros((N, N), dtype=float)

    for post in PLUS_NEURONS:
        for pre in PLUS_NEURONS:

            if post != pre:

                connections[post, pre] = 1
                weights[post, pre] = W_EXC

    return connections, weights


def simulate(
    connections,
    weights,
    stimulation,
    nt,
    rng
):

    voltage = np.full(N, E_REST)

    synaptic_conductance = np.zeros(N)

    spike_state = np.zeros(N, dtype=int)

    spike_count = np.zeros(N, dtype=int)

    refractory_timer = np.zeros(N)

    for t in range(nt):

        new_spike_state = np.zeros(N, dtype=int)

        refractory_timer = np.maximum(
            refractory_timer - DT,
            0
        )

        external_input = np.zeros(N)

        for neuron in stimulation:
            external_input[neuron] = I_EXT

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

    return spike_count, weights


def run_experiment(seed=23):

    connections, weights = create_connections()

    rng = np.random.default_rng(seed)

    voltage = np.full(N, E_REST)

    synaptic_conductance = np.zeros(N)

    spike_state = np.zeros(N, dtype=int)

    spike_count_learning = np.zeros(N, dtype=int)

    refractory_timer = np.zeros(N)

    learning_complete = False

    learning_step = 0

    for t in range(NT_LEARNING):

        new_spike_state = np.zeros(N, dtype=int)

        refractory_timer = np.maximum(
            refractory_timer - DT,
            0
        )

        external_input = np.zeros(N)

        for neuron in LEARNING_STIMULATION:
            external_input[neuron] = I_EXT

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

                spike_count_learning[i] += 1

                refractory_timer[i] = REFRACTORY_PERIOD

        for pre in range(N):

            if new_spike_state[pre] == 1:

                for post in range(N):

                    if connections[post, pre] == 1:

                        weights[post, pre] += PLASTICITY_AMOUNT

        spike_state = new_spike_state

        maximum_weight = np.max(weights)

        if maximum_weight >= WEIGHT_THRESHOLD:

            learning_complete = True

            learning_step = t + 1

            break

    voltage = np.full(N, E_REST)

    synaptic_conductance = np.zeros(N)

    spike_state = np.zeros(N, dtype=int)

    spike_count_recall = np.zeros(N, dtype=int)

    refractory_timer = np.zeros(N)

    for t in range(NT_RECALL):

        new_spike_state = np.zeros(N, dtype=int)

        refractory_timer = np.maximum(
            refractory_timer - DT,
            0
        )

        external_input = np.zeros(N)

        for neuron in RECALL_STIMULATION:
            external_input[neuron] = I_EXT

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

                spike_count_recall[i] += 1

                refractory_timer[i] = REFRACTORY_PERIOD

        spike_state = new_spike_state

    return (
        spike_count_learning,
        spike_count_recall,
        weights,
        learning_complete,
        learning_step
    )


learning_counts, recall_counts, final_weights, learning_complete, learning_step = run_experiment()


print("LEARNING COMPLETE:", learning_complete)

print("LEARNING STEP:", learning_step)

print("\nLEARNING SPIKE COUNTS")

print(learning_counts.reshape(NY, NX))

print("\nRECALL SPIKE COUNTS")

print(recall_counts.reshape(NY, NX))

print("\nFINAL SYNAPTIC WEIGHTS")

print(final_weights)


plt.figure(figsize=(7, 7))

plt.imshow(
    recall_counts.reshape(NY, NX),
    cmap="viridis",
    interpolation="nearest"
)

plt.colorbar(label="Spike Count")

for i in range(NY):

    for j in range(NX):

        plt.text(
            j,
            i,
            str(recall_counts.reshape(NY, NX)[i, j]),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

plt.xticks(range(NX), range(NX))
plt.yticks(range(NY), range(NY))

plt.xlabel("X position")
plt.ylabel("Y position")

plt.title("PLUS Pattern Recall After Synaptic Plasticity")

plt.tight_layout()

plt.show()