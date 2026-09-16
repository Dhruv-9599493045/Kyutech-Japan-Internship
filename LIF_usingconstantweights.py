import numpy as np
import matplotlib.pyplot as plt

NX = 5
NY = 5
N = NX * NY

p1 = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

p2 = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])


TAU = 20.0

W_EXC = 5.0

E_REST = -65.0
THETA = -55.0
R_M = 16.0
I_EXT = 1.5
DECAY = 0.5

NT = 1000
DT = 1.0

def create_connection():

    w = np.zeros((N, N))

    for i in range(N):

        ix = i // NY
        iy = i % NY

        for j in range(N):

            jx = j // NY
            jy = j % NY

            if i != j:

                w[i, j] += (
                    W_EXC
                    * p1[ix, iy]
                    * p1[jx, jy]
                )

                w[i, j] += (
                    W_EXC
                    * p2[ix, iy]
                    * p2[jx, jy]
                )

    return w

def set_input():

    i_ext = np.zeros(N)

    for i in range(NX):

        for j in range(NY):

            index = j + NY * i

            if j < NX // 2:

                i_ext[index] = I_EXT * p1[i, j]

            else:

                i_ext[index] = 0.0

    return i_ext


weights = create_connection()

i_ext = set_input()

v = np.full(N, E_REST)

g_syn = np.zeros(N)


s = np.zeros(N, dtype=bool)


ns = np.zeros(N, dtype=int)


rng = np.random.default_rng(23)


for nt in range(NT):
    for i in range(N):
        r = 0.0
        for j in range(N):
            r += weights[i, j] * s[j]
        g_syn[i] = DECAY * g_syn[i] + r


    for i in range(N):
        random_input = rng.random()
        v[i] += DT * (
            -(v[i] - E_REST)
            + g_syn[i]
            + R_M * (i_ext[i] + random_input)
        ) / TAU

        s[i] = v[i] > THETA

        if s[i]:
            ns[i] += 1
            v[i] = E_REST


spike_matrix = ns.reshape(NX, NY)


print("Spike count matrix:")
print(spike_matrix)

plt.figure(figsize=(7, 6))

plt.imshow(
    spike_matrix,
    cmap="YlOrRd",
    interpolation="nearest"
)

plt.colorbar(label="Number of spikes")

plt.title("Hopfield Network - PLUS Pattern")

plt.xticks(range(NY))
plt.yticks(range(NX))


for i in range(NX):

    for j in range(NY):

        plt.text(
            j,
            i,
            str(spike_matrix[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=13,
            fontweight="bold"
        )


plt.xlabel("Y")
plt.ylabel("X")

plt.tight_layout()

plt.show()

