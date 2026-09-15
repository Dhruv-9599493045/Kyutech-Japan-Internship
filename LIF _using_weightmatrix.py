import numpy as np
import matplotlib.pyplot as plt 

def lif_neuron(I,T=200,dt=0.1):
    v_rest = -65
    v_reset= -65
    v_threshold = -50
    tau_m = 10

    n = int(T/dt)

    t_values = np.zeros(n+1)
    v_values = np.zeros(n+1)

    t_values[0]=0
    v_values[0]= v_reset

    spikes = []

    for k in range(n):
        t = t_values[k]
        v = v_values[k]
        dv = (v_rest -v + I)/tau_m

        v_new = v+dt*dv
        if v_new >= v_threshold:
            spikes.append(t+dt)

            v_new=v_reset

        t_values[k+1] = t + dt 
        v_values[k+1] = v_new
    return t_values,v_values,spikes

weights = np.array([
    [32,0,4,0,31],
    [1,29,4,30,0],
    [7,4,33,6,7],
    [0,145,5,32,1],
    [144,1,5,0,31],
], dtype=float)

rows,cols = weights.shape 

T = 200
dt=0.1

spike_counts = np.zeros((rows,cols))
all_spikes=[]
selected_voltages={}
time_values=None

for i in range(rows):
    for j in range(cols):
        I = weights[i,j]

        t_values,v_values,spikes = lif_neuron(
            I=I,
            T=T,
            dt=dt
        )
        time_values = t_values

        spike_counts[i,j]=len(spikes)
        neuron_index = i*cols+j

        for spike_time in spikes:
            all_spikes.append(
                (spike_time,neuron_index)
            )

        if (i,j) in [
            (0,0),
            (0,4),
            (1,1),
            (3,1),
            (4,0)
        ]:
            selected_voltages[(i,j)] = v_values

spike_times = []
spike_neuron_indices = []
for spike_time,neuron_index in all_spikes:
    spike_times.append(spike_time)
    spike_neuron_indices.append(neuron_index)

output_pattern = (spike_counts>0).astype(int)


#---------
#figure plot
#---------

input_pattern = np.array([
    [1,0,0,0,1],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,1,0,1,0],
    [1,0,0,0,1]
], dtype=float)
fig = plt.figure(figsize=(15,9))

ax1= plt.subplot(2,3,1)
ax1.imshow(
    input_pattern,
    cmap="Reds",
    vmin=0,
    vmax=1
)
ax1.set_title("Input Pattern (5x5 Cross)")
ax1.set_xlabel("Columns")
ax1.set_ylabel("Row")

for i in range(rows):
    ax1.text(
        j,
        i,
        str(input_pattern[i,j]),
        ha="center",
        va="center"
    )

ax2 = plt.subplot(2,3,2)


for index, ((i,j),voltage) in enumerate(selected_voltages.items()):
    ax2.plot(
        time_values,
        voltage,
        label=f"Neuron({i},{j})"
    )
ax2.axhline(
    -50,
    linestyle="--",
    label="Threshold"
)

ax2.set_title("Membrane voltage Of Selected Neurons")
ax2.set_xlabel("Time(ms)")
ax2.set_ylabel("Voltage")
ax2.legend()
ax2.grid(True)

ax3= plt.subplot(2,3,3)
ax3.scatter(
    spike_times,
    spike_neuron_indices,
    s=10
)
ax3.set_title("Spike Raster Plot")
ax3.set_xlabel("Time")
ax3.set_ylabel("Neuron Index")
ax3.grid(True)

ax4 = plt.subplot(2,3,4)
image= ax4.imshow(
    spike_counts,
    cmap="Reds"
)
ax4.set_title("Spike Count")
ax4.set_xlabel("Column")
ax4.set_ylabel("Row")

for i in range(rows):
    for j in range(cols):
        ax4.text(
            j,
            i,
            str(int(spike_counts[i,j])),
            ha="center",
            va="center"
        )
plt.colorbar(image,ax=ax4, label="Spike Count")

output_pattern = (spike_counts>0).astype(int)

ax5=plt.subplot(2,3,5)
ax5.imshow(
    output_pattern,
    cmap="Reds",
    vmin=0,
    vmax=1
)

ax5.set_title("Output Pattern From LIF SPiked")
ax5.set_xlabel("Column")
ax5.set_ylabel("Row")
for i in range(rows):
    for j in range(cols):
        ax5.text(
            j,
            i,
            str(output_pattern[i,j]),
            ha="center",
            va="center"
        )
equation_text="""
LIF Model
dv/dt = (Vrest - v + I)/ tau_m
if v >= Vthreshold:
    spike=1
    v= Vreset
    
parameters:

Vrest = -65
Vreset = -65
Vthreshold = -50
tau_m= 10
dt = 0.1

Active Current = 25
Inactive Current = o
"""
ax6=plt.subplot(2,3,6)
ax6.axis("off")
ax6.text(
    0.05,
    0.95,
    equation_text,
    fontsize=11,
    verticalalignment="top"
)

plt.suptitle(
    "5x5 pattern represented using LIF neurons",
    fontsize = 16
)
plt.tight_layout()
plt.show()
