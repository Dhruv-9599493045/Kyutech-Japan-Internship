import numpy as np
import matplotlib.pyplot as plt
import random

NUM_HOLES = 18

NUM_PLACE_CELLS = 1000
NUM_SOCIAL_PLACE_CELLS = 1000
NUM_DIRECTION_CELLS = 36

TOTAL_NEURONS = (
    NUM_PLACE_CELLS
    + NUM_SOCIAL_PLACE_CELLS
    + NUM_DIRECTION_CELLS
)

DT = 0.1

ARENA_RADIUS = 55.0
HOLE_RADIUS = 47.0
CENTRE_HOLE_RADIUS = 10.0
OUTER_HOLE_RADIUS = 4.5

DEMONSTRATOR_SPEED = 1.5
DIRECTION_CHANGE = 0.35

PLACE_FIELD_WIDTH = 5.0

PLACE_CM = 0.2688
PLACE_RM = 186.0
PLACE_REST = -65.8
PLACE_THRESHOLD = -42.3
PLACE_RESET = -65.8

DIRECTION_CM = 0.1
DIRECTION_RM = 180.0
DIRECTION_REST = -76.0
DIRECTION_THRESHOLD = -44.0
DIRECTION_RESET = -76.0

OBSERVATION_CURRENT = 20.0

OBSERVATION_TIME = 60.0

STEPS_PER_SECOND = 10

DEMONSTRATOR_STEPS = int(
    OBSERVATION_TIME * STEPS_PER_SECOND
)

np.random.seed(23)
random.seed(23)


def lif_step(
    voltage,
    current,
    cm,
    rm,
    rest,
    threshold,
    reset
):

    dv = (
        -(voltage - rest)
        + rm * current
    ) / (rm * cm)

    voltage += DT * dv

    spike = 0

    if voltage >= threshold:

        spike = 1
        voltage = reset

    return voltage, spike


def create_maze():

    angles = np.linspace(
        0,
        2 * np.pi,
        NUM_HOLES,
        endpoint=False
    )

    x = HOLE_RADIUS * np.cos(angles)
    y = HOLE_RADIUS * np.sin(angles)

    return x, y


def create_random_start():

    angle = random.uniform(
        0,
        2 * np.pi
    )

    radius = (
        np.sqrt(
            random.uniform(0, 1)
        )
        * (
            ARENA_RADIUS
            - OUTER_HOLE_RADIUS
        )
    )

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    return x, y


def generate_demonstrator_path(
    start_x,
    start_y,
    number_of_steps
):

    x = start_x
    y = start_y

    heading = random.uniform(
        0,
        2 * np.pi
    )

    path_x = [x]
    path_y = [y]

    for step in range(
        number_of_steps
    ):

        random_turn = np.random.normal(
            0,
            DIRECTION_CHANGE
        )

        heading += random_turn

        dx = (
            DEMONSTRATOR_SPEED
            * np.cos(heading)
        )

        dy = (
            DEMONSTRATOR_SPEED
            * np.sin(heading)
        )

        new_x = x + dx
        new_y = y + dy

        distance_from_centre = np.sqrt(
            new_x ** 2
            + new_y ** 2
        )

        if (
            distance_from_centre
            > ARENA_RADIUS - 2
        ):

            wall_angle = np.arctan2(
                y,
                x
            )

            heading = (
                wall_angle
                + np.pi
            )

            new_x = x + (
                DEMONSTRATOR_SPEED
                * np.cos(heading)
            )

            new_y = y + (
                DEMONSTRATOR_SPEED
                * np.sin(heading)
            )

        x = new_x
        y = new_y

        path_x.append(x)
        path_y.append(y)

    return (
        np.array(path_x),
        np.array(path_y)
    )


def create_place_cells():

    angles = np.random.uniform(
        0,
        2 * np.pi,
        NUM_PLACE_CELLS
    )

    radii = (
        ARENA_RADIUS
        * np.sqrt(
            np.random.uniform(
                0,
                1,
                NUM_PLACE_CELLS
            )
        )
    )

    x = (
        radii
        * np.cos(angles)
    )

    y = (
        radii
        * np.sin(angles)
    )

    return np.column_stack(
        (x, y)
    )


def create_social_place_cells():

    angles = np.random.uniform(
        0,
        2 * np.pi,
        NUM_SOCIAL_PLACE_CELLS
    )

    radii = (
        ARENA_RADIUS
        * np.sqrt(
            np.random.uniform(
                0,
                1,
                NUM_SOCIAL_PLACE_CELLS
            )
        )
    )

    x = (
        radii
        * np.cos(angles)
    )

    y = (
        radii
        * np.sin(angles)
    )

    return np.column_stack(
        (x, y)
    )


def calculate_place_activity(
    rat_x,
    rat_y,
    place_centers
):

    dx = (
        place_centers[:, 0]
        - rat_x
    )

    dy = (
        place_centers[:, 1]
        - rat_y
    )

    distance_squared = (
        dx ** 2
        + dy ** 2
    )

    activity = np.exp(
        -distance_squared
        / (
            2
            * PLACE_FIELD_WIDTH ** 2
        )
    )

    return activity


def observe_demonstrator(
    demonstrator_x,
    demonstrator_y,
    place_centers,
    social_centers
):

    place_voltage = np.full(
        NUM_PLACE_CELLS,
        PLACE_REST
    )

    social_voltage = np.full(
        NUM_SOCIAL_PLACE_CELLS,
        PLACE_REST
    )

    direction_voltage = np.full(
        NUM_DIRECTION_CELLS,
        DIRECTION_REST
    )

    place_spike_history = []

    social_spike_history = []

    direction_spike_history = []

    firing_positions = []

    observed_positions = []

    for rat_x, rat_y in zip(
        demonstrator_x,
        demonstrator_y
    ):

        place_activity = (
            calculate_place_activity(
                rat_x,
                rat_y,
                place_centers
            )
        )

        social_activity = (
            calculate_place_activity(
                rat_x,
                rat_y,
                social_centers
            )
        )

        place_spikes = np.zeros(
            NUM_PLACE_CELLS,
            dtype=int
        )

        social_spikes = np.zeros(
            NUM_SOCIAL_PLACE_CELLS,
            dtype=int
        )

        for i in range(
            NUM_PLACE_CELLS
        ):

            (
                place_voltage[i],
                place_spikes[i]
            ) = lif_step(
                place_voltage[i],
                place_activity[i]
                * OBSERVATION_CURRENT,
                PLACE_CM,
                PLACE_RM,
                PLACE_REST,
                PLACE_THRESHOLD,
                PLACE_RESET
            )

        for i in range(
            NUM_SOCIAL_PLACE_CELLS
        ):

            (
                social_voltage[i],
                social_spikes[i]
            ) = lif_step(
                social_voltage[i],
                social_activity[i]
                * OBSERVATION_CURRENT,
                PLACE_CM,
                PLACE_RM,
                PLACE_REST,
                PLACE_THRESHOLD,
                PLACE_RESET
            )

        active_place_cells = np.where(
            place_spikes == 1
        )[0]

        if len(active_place_cells) > 0:

            firing_positions.extend(
                place_centers[
                    active_place_cells
                ]
            )

        direction_activity = np.zeros(
            NUM_DIRECTION_CELLS
        )

        movement_angle = np.arctan2(
            rat_y,
            rat_x
        )

        direction_normalized = (
            movement_angle + np.pi
        ) / (
            2 * np.pi
        )

        direction_index = int(
            direction_normalized
            * NUM_DIRECTION_CELLS
        )

        direction_index = min(
            direction_index,
            NUM_DIRECTION_CELLS - 1
        )

        direction_activity[
            direction_index
        ] = 1.0

        direction_spikes = np.zeros(
            NUM_DIRECTION_CELLS,
            dtype=int
        )

        for i in range(
            NUM_DIRECTION_CELLS
        ):

            (
                direction_voltage[i],
                direction_spikes[i]
            ) = lif_step(
                direction_voltage[i],
                direction_activity[i]
                * OBSERVATION_CURRENT,
                DIRECTION_CM,
                DIRECTION_RM,
                DIRECTION_REST,
                DIRECTION_THRESHOLD,
                DIRECTION_RESET
            )

        place_spike_history.append(
            place_spikes
        )

        social_spike_history.append(
            social_spikes
        )

        direction_spike_history.append(
            direction_spikes
        )

        observed_positions.append(
            (rat_x, rat_y)
        )

    return (
        np.array(place_spike_history),
        np.array(social_spike_history),
        np.array(direction_spike_history),
        np.array(observed_positions),
        np.array(firing_positions)
    )


def find_nearest_hole(
    x,
    y,
    hole_x,
    hole_y
):

    distances = np.sqrt(
        (hole_x - x) ** 2
        + (hole_y - y) ** 2
    )

    return np.argmin(
        distances
    )


def calculate_shortest_path(
    goal_hole,
    hole_x,
    hole_y
):

    goal_x = hole_x[goal_hole]
    goal_y = hole_y[goal_hole]

    observer_x = np.linspace(
        0,
        goal_x,
        150
    )

    observer_y = np.linspace(
        0,
        goal_y,
        150
    )

    return (
        observer_x,
        observer_y
    )


def plot_demonstrator_trajectory(
    hole_x,
    hole_y,
    demonstrator_x,
    demonstrator_y,
    start_x,
    start_y,
    goal_hole
):

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    arena = plt.Circle(
        (0, 0),
        ARENA_RADIUS,
        fill=False,
        color="black",
        linewidth=2
    )

    ax.add_patch(
        arena
    )

    for i in range(
        NUM_HOLES
    ):

        if i == goal_hole:

            circle = plt.Circle(
                (
                    hole_x[i],
                    hole_y[i]
                ),
                OUTER_HOLE_RADIUS,
                color="red",
                alpha=0.85
            )

        else:

            circle = plt.Circle(
                (
                    hole_x[i],
                    hole_y[i]
                ),
                OUTER_HOLE_RADIUS,
                color="gray",
                alpha=0.85
            )

        ax.add_patch(
            circle
        )

    centre = plt.Circle(
        (0, 0),
        CENTRE_HOLE_RADIUS,
        color="lightgray",
        alpha=0.9
    )

    ax.add_patch(
        centre
    )

    ax.plot(
        demonstrator_x,
        demonstrator_y,
        color="red",
        linewidth=1.2
    )

    ax.scatter(
        start_x,
        start_y,
        s=120,
        color="black",
        zorder=5
    )

    ax.text(
        start_x + 2,
        start_y + 2,
        "START",
        fontsize=10
    )

    ax.text(
        hole_x[goal_hole] + 2,
        hole_y[goal_hole] + 5,
        "GOAL",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlim(
        -60,
        60
    )

    ax.set_ylim(
        -60,
        60
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "X [cm]",
        fontsize=13
    )

    ax.set_ylabel(
        "Y [cm]",
        fontsize=13
    )

    ax.set_title(
        "Demonstrator trajectory",
        fontsize=15
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


def plot_spatial_neuron_firing(
    hole_x,
    hole_y,
    firing_positions,
    goal_hole
):

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    arena = plt.Circle(
        (0, 0),
        ARENA_RADIUS,
        fill=False,
        color="black",
        linewidth=2
    )

    ax.add_patch(
        arena
    )

    for i in range(
        NUM_HOLES
    ):

        if i == goal_hole:

            circle = plt.Circle(
                (
                    hole_x[i],
                    hole_y[i]
                ),
                OUTER_HOLE_RADIUS,
                color="red",
                alpha=0.85
            )

        else:

            circle = plt.Circle(
                (
                    hole_x[i],
                    hole_y[i]
                ),
                OUTER_HOLE_RADIUS,
                color="gray",
                alpha=0.85
            )

        ax.add_patch(
            circle
        )

    centre = plt.Circle(
        (0, 0),
        CENTRE_HOLE_RADIUS,
        color="lightgray",
        alpha=0.9
    )

    ax.add_patch(
        centre
    )

    if len(firing_positions) > 0:

        firing_positions = np.array(
            firing_positions
        )

        ax.scatter(
            firing_positions[:, 0],
            firing_positions[:, 1],
            s=5,
            color="red",
            alpha=0.65
        )

    ax.set_xlim(
        -60,
        60
    )

    ax.set_ylim(
        -60,
        60
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "X [cm]",
        fontsize=13
    )

    ax.set_ylabel(
        "Y [cm]",
        fontsize=13
    )

    ax.set_title(
        "Spatial neuron firing",
        fontsize=15
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


def plot_observer(
    hole_x,
    hole_y,
    goal_hole,
    observer_x,
    observer_y
):

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    arena = plt.Circle(
        (0, 0),
        ARENA_RADIUS,
        fill=False,
        color="black",
        linewidth=2
    )

    ax.add_patch(
        arena
    )

    for i in range(
        NUM_HOLES
    ):

        if i == goal_hole:

            circle = plt.Circle(
                (
                    hole_x[i],
                    hole_y[i]
                ),
                OUTER_HOLE_RADIUS,
                color="red",
                alpha=0.85
            )

        else:

            circle = plt.Circle(
                (
                    hole_x[i],
                    hole_y[i]
                ),
                OUTER_HOLE_RADIUS,
                color="gray",
                alpha=0.85
            )

        ax.add_patch(
            circle
        )

    centre = plt.Circle(
        (0, 0),
        CENTRE_HOLE_RADIUS,
        color="lightgray",
        alpha=0.9
    )

    ax.add_patch(
        centre
    )

    ax.plot(
        observer_x,
        observer_y,
        color="blue",
        linewidth=3
    )

    ax.scatter(
        0,
        0,
        s=120,
        color="black"
    )

    ax.scatter(
        hole_x[goal_hole],
        hole_y[goal_hole],
        s=250,
        marker="*",
        color="red"
    )

    ax.set_xlim(
        -60,
        60
    )

    ax.set_ylim(
        -60,
        60
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "X [cm]",
        fontsize=13
    )

    ax.set_ylabel(
        "Y [cm]",
        fontsize=13
    )

    ax.set_title(
        "Observer goal-directed trajectory",
        fontsize=15
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


hole_x, hole_y = create_maze()

goal_hole = random.randint(
    0,
    NUM_HOLES - 1
)

start_x, start_y = (
    create_random_start()
)

print()
print(
    "=========================================="
)

print(
    "2036-NEURON OBSERVATIONAL LEARNING"
)

print(
    "WITHOUT STDP"
)

print(
    "=========================================="
)

print()

print(
    "Place neurons:",
    NUM_PLACE_CELLS
)

print(
    "Social-place neurons:",
    NUM_SOCIAL_PLACE_CELLS
)

print(
    "Direction neurons:",
    NUM_DIRECTION_CELLS
)

print(
    "Total neurons:",
    TOTAL_NEURONS
)

print()

print(
    "Demonstrator random start:"
)

print(
    "X =",
    round(start_x, 2),
    "cm"
)

print(
    "Y =",
    round(start_y, 2),
    "cm"
)

print()

print(
    "Goal hole:",
    goal_hole
)

print()

print(
    "Generating demonstrator trajectory..."
)

demonstrator_x, demonstrator_y = (
    generate_demonstrator_path(
        start_x,
        start_y,
        DEMONSTRATOR_STEPS
    )
)

place_centers = (
    create_place_cells()
)

social_centers = (
    create_social_place_cells()
)

print(
    "Demonstrator positions:",
    len(demonstrator_x)
)

print()

print(
    "Observing demonstrator..."
)

(
    place_spikes,
    social_spikes,
    direction_spikes,
    observed_positions,
    firing_positions
) = observe_demonstrator(
    demonstrator_x,
    demonstrator_y,
    place_centers,
    social_centers
)

last_x = demonstrator_x[-1]
last_y = demonstrator_y[-1]

goal_hole = find_nearest_hole(
    last_x,
    last_y,
    hole_x,
    hole_y
)

print(
    "Identified goal:",
    goal_hole
)

print(
    "Spatial neuron firing events:",
    len(firing_positions)
)

observer_x, observer_y = (
    calculate_shortest_path(
        goal_hole,
        hole_x,
        hole_y
    )
)

print()

print(
    "Observer starts from centre."
)

print(
    "Observer travels directly to goal."
)

print()

print(
    "Total place-cell spikes:",
    int(
        np.sum(
            place_spikes
        )
    )
)

print(
    "Total social-place-cell spikes:",
    int(
        np.sum(
            social_spikes
        )
    )
)

print(
    "Total direction-cell spikes:",
    int(
        np.sum(
            direction_spikes
        )
    )
)

plot_demonstrator_trajectory(
    hole_x,
    hole_y,
    demonstrator_x,
    demonstrator_y,
    start_x,
    start_y,
    goal_hole
)

plot_spatial_neuron_firing(
    hole_x,
    hole_y,
    firing_positions,
    goal_hole
)

plot_observer(
    hole_x,
    hole_y,
    goal_hole,
    observer_x,
    observer_y
)