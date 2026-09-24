import numpy as np
import matplotlib.pyplot as plt

NX = 5
NY = 5
N = NX * NY

W_EXC = 0.5
W_CROSS = 0.02

PLASTICITY_AMOUNT = 0.1
CROSS_PLASTICITY_AMOUNT = 0.005

WEIGHT_THRESHOLD = 5.0

I_EXT = 3.0

PLUS_RECALL_INPUT = 2.2
CROSS_RECALL_INPUT = 1.5

DECAY = 0.5

NT_LEARNING = 10000
NT_RECALL = 5000

DT = 0.1

TAU = 20.0
E_REST = -65.0
THETA = -55.0
R_M = 16.0

REFRACTORY_PERIOD = 2.0


PLUS_NEURONS = [
    2, 7,
    10, 11, 12, 13, 14,
    17, 22
]

CROSS_NEURONS = [
    0, 4,
    6, 8,
    12,
    16, 18,
    20, 24
]

PLUS_RECALL = [10, 11]

CROSS_RECALL = [20, 16]


def create_connections():

    connections = np.zeros((N, N), dtype=int)
    weights = np.zeros((N, N), dtype=float)

    for post in range(N):

        for pre in range(N):

            if post == pre:
                continue

            post_plus = post in PLUS_NEURONS
            pre_plus = pre in PLUS_NEURONS

            post_cross = post in CROSS_NEURONS
            pre_cross = pre in CROSS_NEURONS

            if (post_plus and pre_plus) or (post_cross and pre_cross):

                connections[post, pre] = 1
                weights[post, pre] = W_EXC

            elif (post_plus and pre_cross) or (post_cross and pre_plus):

                connections[post, pre] = 1
                weights[post, pre] = W_CROSS

    return connections, weights


def learn_pattern(
    stimulation_neurons,
    connections,
    weights,
    seed
):

    voltage = np.full(N, E_REST)

    synaptic_conductance = np.zeros(N)

    spike_state = np.zeros(N, dtype=int)

    refractory_timer = np.zeros(N)

    rng = np.random.default_rng(seed)

    learning_step = 0

    for t in range(NT_LEARNING):

        new_spike_state = np.zeros(N, dtype=int)

        refractory_timer = np.maximum(
            refractory_timer - DT,
            0
        )

        for i in range(N):

            recurrent_input = np.sum(
                connections[i, :]
                * weights[i, :]
                * spike_state
            )

            synaptic_conductance[i] = (
                DECAY * synaptic_conductance[i]
                + recurrent_input
            )

        for i in range(N):

            if refractory_timer[i] > 0:

                voltage[i] = E_REST

                continue

            external_input = 0.0

            if i in stimulation_neurons:

                external_input = I_EXT

            random_input = rng.random()

            total_input = (
                external_input
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

                refractory_timer[i] = REFRACTORY_PERIOD

        for pre in range(N):

            if new_spike_state[pre] == 1:

                for post in range(N):

                    if connections[post, pre] == 1:

                        if (
                            post in stimulation_neurons
                            and pre in stimulation_neurons
                        ):

                            weights[post, pre] += (
                                PLASTICITY_AMOUNT
                            )

                        else:

                            weights[post, pre] += (
                                CROSS_PLASTICITY_AMOUNT
                            )

        spike_state = new_spike_state

        pattern_weights = []

        for post in stimulation_neurons:

            for pre in stimulation_neurons:

                if connections[post, pre] == 1:

                    pattern_weights.append(
                        weights[post, pre]
                    )

        if len(pattern_weights) > 0:

            if max(pattern_weights) >= WEIGHT_THRESHOLD:

                learning_step = t + 1

                break

    return weights, learning_step


def recall_pattern(
    stimulation_neurons,
    connections,
    weights,
    seed,
    recall_input
):

    voltage = np.full(N, E_REST)

    synaptic_conductance = np.zeros(N)

    spike_state = np.zeros(N, dtype=int)

    spike_count = np.zeros(N, dtype=int)

    refractory_timer = np.zeros(N)

    rng = np.random.default_rng(seed)

    for t in range(NT_RECALL):

        new_spike_state = np.zeros(N, dtype=int)

        refractory_timer = np.maximum(
            refractory_timer - DT,
            0
        )

        for i in range(N):

            recurrent_input = np.sum(
                connections[i, :]
                * weights[i, :]
                * spike_state
            )

            synaptic_conductance[i] = (
                DECAY * synaptic_conductance[i]
                + recurrent_input
            )

        for i in range(N):

            if refractory_timer[i] > 0:

                voltage[i] = E_REST

                continue

            external_input = 0.0

            if i in stimulation_neurons:

                external_input = recall_input

            random_input = rng.random()

            total_input = (
                external_input
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

        spike_state = new_spike_state

    return spike_count.reshape(NY, NX)


def plot_result(result, title):

    plt.figure(figsize=(7, 7))

    plt.imshow(
        result,
        interpolation="nearest"
    )

    plt.colorbar(label="Spike count")

    plt.xticks(range(NX))
    plt.yticks(range(NY))

    for i in range(NY):

        for j in range(NX):

            plt.text(
                j,
                i,
                str(result[i, j]),
                ha="center",
                va="center"
            )

    plt.title(title)

    plt.xlabel("Neuron X")

    plt.ylabel("Neuron Y")

    plt.tight_layout()

    plt.show()


connections, weights = create_connections()


print("Learning PLUS pattern...")

weights, plus_learning_step = learn_pattern(
    PLUS_NEURONS,
    connections,
    weights,
    seed=23
)

print(
    "PLUS learning completed at step:",
    plus_learning_step
)


print("\nLearning CROSS pattern...")

weights, cross_learning_step = learn_pattern(
    CROSS_NEURONS,
    connections,
    weights,
    seed=24
)

print(
    "CROSS learning completed at step:",
    cross_learning_step
)


print("\nPLUS RECALL")

plus_recall = recall_pattern(
    PLUS_RECALL,
    connections,
    weights,
    seed=25,
    recall_input=PLUS_RECALL_INPUT
)

print(plus_recall)

plot_result(
    plus_recall,
    "PLUS Recall"
)


print("\nCROSS RECALL")

cross_recall = recall_pattern(
    CROSS_RECALL,
    connections,
    weights,
    seed=26,
    recall_input=CROSS_RECALL_INPUT
)

print(cross_recall)

plot_result(
    cross_recall,
    "CROSS Recall"
)