
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

OBSERVATION_TIME = 20.0

OBSERVATION_STEPS = int(
    OBSERVATION_TIME / DT
)


ARENA_RADIUS = 55.0
HOLE_RADIUS = 47.0

CENTRE_HOLE_RADIUS = 10.0
OUTER_HOLE_RADIUS = 4.5


DEMONSTRATOR_SPEED = 1.5
OBSERVER_SPEED = 1.2

DEMONSTRATOR_TURN = 0.30
OBSERVER_TURN = 0.22

RANDOM_MOVEMENT = 0.35

GOAL_BIAS = 0.08

LEARNED_GOAL_THRESHOLD = 0.85
DEMONSTRATOR_GOAL_TIME = 95.0

HOLE_RECOGNITION_DISTANCE = 5.5


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


np.random.seed()
random.seed()


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
    ) / (
        rm * cm
    )

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

    hole_x = (
        HOLE_RADIUS
        * np.cos(angles)
    )

    hole_y = (
        HOLE_RADIUS
        * np.sin(angles)
    )

    return hole_x, hole_y


def create_random_position():

    while True:

        x = np.random.uniform(
            -ARENA_RADIUS + 5,
            ARENA_RADIUS - 5
        )

        y = np.random.uniform(
            -ARENA_RADIUS + 5,
            ARENA_RADIUS - 5
        )

        distance = np.sqrt(
            x ** 2 + y ** 2
        )

        if (
            distance <= ARENA_RADIUS - 5
            and distance > 8
        ):

            return x, y


def generate_demonstrator_path(
    start_x,
    start_y,
    goal_hole,
    hole_x,
    hole_y
):

    x = start_x
    y = start_y

    heading = np.random.uniform(
        0,
        2 * np.pi
    )

    goal_x = hole_x[goal_hole]
    goal_y = hole_y[goal_hole]

    path_x = [x]
    path_y = [y]

    total_steps = int(
        140.0 / DT
    )

    for step in range(
        total_steps
    ):

        time = step * DT

        dx_goal = (
            goal_x - x
        )

        dy_goal = (
            goal_y - y
        )

        distance_to_goal = np.sqrt(
            dx_goal ** 2
            + dy_goal ** 2
        )

        if distance_to_goal <= HOLE_RECOGNITION_DISTANCE:

            x = goal_x
            y = goal_y

            path_x.append(x)
            path_y.append(y)

            break

        if time < 60.0:

            heading += np.random.normal(
                0,
                DEMONSTRATOR_TURN
            )

            movement_speed = DEMONSTRATOR_SPEED

        else:

            target_angle = np.arctan2(
                dy_goal,
                dx_goal
            )

            angle_difference = (
                target_angle - heading
            )

            angle_difference = np.arctan2(
                np.sin(
                    angle_difference
                ),
                np.cos(
                    angle_difference
                )
            )

            if time < 80.0:

                heading += (
                    0.20
                    * angle_difference
                )

                heading += np.random.normal(
                    0,
                    0.12
                )

                movement_speed = DEMONSTRATOR_SPEED

            else:

                heading = target_angle

                remaining_time = max(
                    DEMONSTRATOR_GOAL_TIME - time,
                    DT
                )

                required_speed = (
                    distance_to_goal
                    / remaining_time
                )

                movement_speed = max(
                    DEMONSTRATOR_SPEED,
                    required_speed
                )

        new_x = (
            x
            + movement_speed
            * np.cos(heading)
            * DT
        )

        new_y = (
            y
            + movement_speed
            * np.sin(heading)
            * DT
        )

        distance = np.sqrt(
            new_x ** 2
            + new_y ** 2
        )

        if distance >= (
            ARENA_RADIUS - 2
        ):

            wall_angle = np.arctan2(
                y,
                x
            )

            heading = (
                wall_angle
                + np.pi
                + np.random.normal(
                    0,
                    0.2
                )
            )

            new_x = (
                x
                + DEMONSTRATOR_SPEED
                * np.cos(heading)
                * DT
            )

            new_y = (
                y
                + DEMONSTRATOR_SPEED
                * np.sin(heading)
                * DT
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

    place_spikes_history = []

    social_spikes_history = []

    direction_spikes_history = []

    firing_positions = []

    for k in range(
        len(demonstrator_x)
    ):

        rat_x = demonstrator_x[k]
        rat_y = demonstrator_y[k]

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

        active_cells = np.where(
            place_spikes == 1
        )[0]

        if len(active_cells) > 0:

            firing_positions.extend(
                place_centers[
                    active_cells
                ]
            )

        if k > 0:

            dx = (
                demonstrator_x[k]
                - demonstrator_x[k - 1]
            )

            dy = (
                demonstrator_y[k]
                - demonstrator_y[k - 1]
            )

            movement_angle = np.arctan2(
                dy,
                dx
            )

        else:

            movement_angle = 0.0

        normalized_angle = (
            movement_angle + np.pi
        ) / (
            2 * np.pi
        )

        direction_index = int(
            normalized_angle
            * NUM_DIRECTION_CELLS
        )

        direction_index = min(
            direction_index,
            NUM_DIRECTION_CELLS - 1
        )

        direction_activity = np.zeros(
            NUM_DIRECTION_CELLS
        )

        direction_activity[
            direction_index
        ] = 1

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

        place_spikes_history.append(
            place_spikes
        )

        social_spikes_history.append(
            social_spikes
        )

        direction_spikes_history.append(
            direction_spikes
        )

    return (
        np.array(
            place_spikes_history
        ),
        np.array(
            social_spikes_history
        ),
        np.array(
            direction_spikes_history
        ),
        np.array(
            firing_positions
        )
    )


def calculate_goal_evidence(
    demonstrator_x,
    demonstrator_y,
    hole_x,
    hole_y
):

    scores = np.zeros(
        NUM_HOLES
    )

    for i in range(
        NUM_HOLES
    ):

        dx = (
            demonstrator_x
            - hole_x[i]
        )

        dy = (
            demonstrator_y
            - hole_y[i]
        )

        distances = np.sqrt(
            dx ** 2
            + dy ** 2
        )

        close_positions = (
            distances < 12
        )

        if np.any(
            close_positions
        ):

            minimum_distance = np.min(
                distances
            )

            scores[i] += (
                1
                / (
                    minimum_distance
                    + 1
                )
            )

    if np.max(scores) > 0:

        scores /= np.max(
            scores
        )

    return scores


def get_learned_goal(
    demonstrator_x,
    demonstrator_y,
    hole_x,
    hole_y,
    goal_evidence
):

    for hole in range(
        NUM_HOLES
    ):

        dx = (
            demonstrator_x
            - hole_x[hole]
        )

        dy = (
            demonstrator_y
            - hole_y[hole]
        )

        distances = np.sqrt(
            dx ** 2
            + dy ** 2
        )

        if len(distances) == 0:
            continue

        if np.min(distances) <= HOLE_RECOGNITION_DISTANCE:

            closest_index = int(
                np.argmin(distances)
            )

            if closest_index == len(distances) - 1:

                return hole

    return None

def get_goal_direction(
    x,
    y,
    hole_x,
    hole_y,
    goal_evidence
):

    weighted_x = 0.0
    weighted_y = 0.0

    total_weight = 0.0

    for i in range(
        NUM_HOLES
    ):

        weight = (
            goal_evidence[i] ** 3
        )

        if weight <= 0:

            continue

        dx = (
            hole_x[i] - x
        )

        dy = (
            hole_y[i] - y
        )

        distance = np.sqrt(
            dx ** 2
            + dy ** 2
        )

        if distance > 0:

            weighted_x += (
                weight
                * dx
                / distance
            )

            weighted_y += (
                weight
                * dy
                / distance
            )

            total_weight += weight

    if total_weight == 0:

        return None

    weighted_x /= total_weight

    weighted_y /= total_weight

    return np.arctan2(
        weighted_y,
        weighted_x
    )


def run_observer(
    hole_x,
    hole_y,
    goal_hole,
    learned_goal
):

    x = 0.0
    y = 0.0

    heading = np.random.uniform(
        0,
        2 * np.pi
    )

    observer_x = [x]
    observer_y = [y]

    visited_holes = []

    search_steps = 0

    if learned_goal is not None:

        print(
            "Observer learned goal hole:",
            learned_goal
        )

        print(
            "Observer will take the shortest direct path."
        )

    else:

        print(
            "Observer did not identify the goal."
        )

        print(
            "Observer will explore the entire maze."
        )

    target_x = None
    target_y = None

    visited_positions = []

    while True:

        search_steps += 1

        current_time = (
            search_steps * DT
        )

        distance_to_actual_goal = np.sqrt(
            (
                x - hole_x[goal_hole]
            ) ** 2
            +
            (
                y - hole_y[goal_hole]
            ) ** 2
        )

        if distance_to_actual_goal <= HOLE_RECOGNITION_DISTANCE:

            x = hole_x[goal_hole]
            y = hole_y[goal_hole]

            observer_x.append(x)
            observer_y.append(y)

            if goal_hole not in visited_holes:
                visited_holes.append(goal_hole)

            print(
                "Observer reached hole:",
                goal_hole,
                "at",
                round(current_time, 2),
                "seconds"
            )

            print()
            print("GOAL FOUND!")

            return (
                np.array(observer_x),
                np.array(observer_y),
                current_time,
                visited_holes
            )

        if learned_goal is not None:

            target_x = hole_x[learned_goal]
            target_y = hole_y[learned_goal]

            target_angle = np.arctan2(
                target_y - y,
                target_x - x
            )

            heading = target_angle

        else:

            if (
                target_x is None
                or
                np.sqrt(
                    (target_x - x) ** 2
                    +
                    (target_y - y) ** 2
                ) < 6.0
            ):

                candidates = []

                for _ in range(100):

                    candidate_x, candidate_y = (
                        create_random_position()
                    )

                    candidate_distance = np.sqrt(
                        (candidate_x - x) ** 2
                        +
                        (candidate_y - y) ** 2
                    )

                    if candidate_distance < 12:
                        continue

                    memory_distance = float("inf")

                    if len(visited_positions) > 0:

                        memory = np.asarray(
                            visited_positions,
                            dtype=float
                        )

                        distances = np.sqrt(
                            (
                                memory[:, 0]
                                - candidate_x
                            ) ** 2
                            +
                            (
                                memory[:, 1]
                                - candidate_y
                            ) ** 2
                        )

                        memory_distance = np.min(
                            distances
                        )

                    score = (
                        0.65
                        * candidate_distance
                        +
                        0.35
                        * memory_distance
                    )

                    candidates.append(
                        (
                            score,
                            candidate_x,
                            candidate_y
                        )
                    )

                if len(candidates) > 0:

                    candidates.sort(
                        reverse=True
                    )

                    target_x = candidates[0][1]
                    target_y = candidates[0][2]

                else:

                    target_x, target_y = (
                        create_random_position()
                    )

            target_angle = np.arctan2(
                target_y - y,
                target_x - x
            )

            angle_difference = (
                target_angle - heading
            )

            angle_difference = np.arctan2(
                np.sin(angle_difference),
                np.cos(angle_difference)
            )

            heading += (
                OBSERVER_TURN
                * angle_difference
            )

            heading += np.random.normal(
                0,
                RANDOM_MOVEMENT
            )

        new_x = (
            x
            + OBSERVER_SPEED
            * np.cos(heading)
            * DT
        )

        new_y = (
            y
            + OBSERVER_SPEED
            * np.sin(heading)
            * DT
        )

        distance_from_center = np.sqrt(
            new_x ** 2
            + new_y ** 2
        )

        if distance_from_center >= (
            ARENA_RADIUS - 2
        ):

            wall_angle = np.arctan2(
                y,
                x
            )

            heading = (
                wall_angle
                + np.pi
                + np.random.normal(
                    0,
                    0.4
                )
            )

            new_x = (
                x
                + OBSERVER_SPEED
                * np.cos(heading)
                * DT
            )

            new_y = (
                y
                + OBSERVER_SPEED
                * np.sin(heading)
                * DT
            )

        x = new_x
        y = new_y

        observer_x.append(x)
        observer_y.append(y)

        visited_positions.append(
            (x, y)
        )

        closest_hole = -1
        closest_distance = float("inf")

        for hole in range(
            NUM_HOLES
        ):

            dx = x - hole_x[hole]
            dy = y - hole_y[hole]

            hole_distance = np.sqrt(
                dx ** 2
                + dy ** 2
            )

            if hole_distance < closest_distance:

                closest_distance = hole_distance
                closest_hole = hole

        if closest_distance <= HOLE_RECOGNITION_DISTANCE:

            if closest_hole not in visited_holes:

                visited_holes.append(
                    closest_hole
                )

                print(
                    "Observer reached hole:",
                    closest_hole,
                    "at",
                    round(current_time, 2),
                    "seconds"
                )

            target_x = None
            target_y = None

            if learned_goal is not None and closest_hole == learned_goal:

                if closest_hole == goal_hole:

                    print()
                    print("GOAL FOUND!")

                    return (
                        np.array(observer_x),
                        np.array(observer_y),
                        current_time,
                        visited_holes
                    )

                heading += np.pi


def plot_demonstrator(
    hole_x,
    hole_y,
    goal_hole,
    start_x,
    start_y,
    demonstrator_x,
    demonstrator_y
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

        color = (
            "red"
            if i == goal_hole
            else "gray"
        )

        hole = plt.Circle(
            (
                hole_x[i],
                hole_y[i]
            ),
            OUTER_HOLE_RADIUS,
            color=color,
            alpha=0.85
        )

        ax.add_patch(
            hole
        )

        ax.text(
            hole_x[i],
            hole_y[i],
            str(i),
            ha="center",
            va="center"
        )

    centre = plt.Circle(
        (0, 0),
        CENTRE_HOLE_RADIUS,
        color="lightgray"
    )

    ax.add_patch(
        centre
    )

    ax.plot(
        demonstrator_x,
        demonstrator_y,
        color="red",
        linewidth=1.5,
        label="Demonstrator"
    )

    ax.scatter(
        start_x,
        start_y,
        color="black",
        s=100,
        zorder=5,
        label="Demonstrator start"
    )

    ax.scatter(
        0,
        0,
        color="blue",
        s=100,
        zorder=5,
        label="Observer start"
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
        "X [cm]"
    )

    ax.set_ylabel(
        "Y [cm]"
    )

    ax.set_title(
        "Demonstrator trajectory — " + str(int(OBSERVATION_TIME)) + " second observation"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


def plot_spatial_activity(
    hole_x,
    hole_y,
    goal_hole,
    firing_positions
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

        color = (
            "red"
            if i == goal_hole
            else "gray"
        )

        hole = plt.Circle(
            (
                hole_x[i],
                hole_y[i]
            ),
            OUTER_HOLE_RADIUS,
            color=color,
            alpha=0.85
        )

        ax.add_patch(
            hole
        )

    centre = plt.Circle(
        (0, 0),
        CENTRE_HOLE_RADIUS,
        color="lightgray"
    )

    ax.add_patch(
        centre
    )

    if len(
        firing_positions
    ) > 0:

        firing_positions = np.array(
            firing_positions
        )

        ax.scatter(
            firing_positions[:, 0],
            firing_positions[:, 1],
            s=5,
            color="red",
            alpha=0.55
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
        "X [cm]"
    )

    ax.set_ylabel(
        "Y [cm]"
    )

    ax.set_title(
        "Spatial neurons during " + str(int(OBSERVATION_TIME)) + " second observation"
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
    observer_y,
    visited_holes,
    search_time
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

            color = "red"

        elif i in visited_holes:

            color = "orange"

        else:

            color = "gray"

        hole = plt.Circle(
            (
                hole_x[i],
                hole_y[i]
            ),
            OUTER_HOLE_RADIUS,
            color=color,
            alpha=0.85
        )

        ax.add_patch(
            hole
        )

        ax.text(
            hole_x[i],
            hole_y[i],
            str(i),
            ha="center",
            va="center"
        )

    centre = plt.Circle(
        (0, 0),
        CENTRE_HOLE_RADIUS,
        color="lightgray"
    )

    ax.add_patch(
        centre
    )

    ax.plot(
        observer_x,
        observer_y,
        color="blue",
        linewidth=2,
        label="Observer"
    )

    ax.scatter(
        0,
        0,
        color="black",
        s=100,
        zorder=5,
        label="Observer start"
    )

    ax.scatter(
        hole_x[goal_hole],
        hole_y[goal_hole],
        color="red",
        marker="*",
        s=250,
        zorder=5,
        label="Goal"
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
        "X [cm]"
    )

    ax.set_ylabel(
        "Y [cm]"
    )

    ax.set_title(
        "Observer trajectory\n"
        + str(int(OBSERVATION_TIME)) + " second observation\n"
        "Search time: "
        + str(
            round(
                search_time,
                2
            )
        )
        + " seconds"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


hole_x, hole_y = create_maze()

goal_hole = random.randrange(
    NUM_HOLES
)

start_x, start_y = (
    create_random_position()
)


print()
print(
    "=========================================="
)
print(
    "OBSERVATIONAL LEARNING EXPERIMENT"
)
print(
    "20 / 60 / 100 / 140 SECOND CONDITIONS"
)
print(
    "LIF MODEL — NO STDP"
)
print(
    "=========================================="
)

print()

print(
    "Total neurons:",
    TOTAL_NEURONS
)

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

print()

print(
    "Random goal hole:",
    goal_hole
)

print(
    "Random demonstrator start:",
    round(start_x, 2),
    round(start_y, 2)
)

print()

print(
    "Generating demonstrator trajectory..."
)

demonstrator_x, demonstrator_y = (
    generate_demonstrator_path(
        start_x,
        start_y,
        goal_hole,
        hole_x,
        hole_y
    )
)


place_centers = (
    create_place_cells()
)

social_centers = (
    create_social_place_cells()
)


observation_times = [
    20.0,
    60.0,
    100.0,
    140.0
]


results = []


for observation_time in observation_times:

    OBSERVATION_TIME = observation_time

    OBSERVATION_STEPS = int(
        OBSERVATION_TIME / DT
    )

    observed_x = (
        demonstrator_x[
            :OBSERVATION_STEPS
        ]
    )

    observed_y = (
        demonstrator_y[
            :OBSERVATION_STEPS
        ]
    )

    print()
    print(
        "=========================================="
    )
    print(
        str(int(OBSERVATION_TIME))
        + " SECOND CONDITION"
    )
    print(
        "=========================================="
    )

    print()

    print(
        "Observer watches demonstrator for:",
        OBSERVATION_TIME,
        "seconds"
    )

    print()

    print(
        "Running spatial neurons..."
    )

    (
        place_spikes,
        social_spikes,
        direction_spikes,
        firing_positions
    ) = observe_demonstrator(
        observed_x,
        observed_y,
        place_centers,
        social_centers
    )

    print(
        "Place-cell spikes:",
        int(
            np.sum(
                place_spikes
            )
        )
    )

    print(
        "Social-place spikes:",
        int(
            np.sum(
                social_spikes
            )
        )
    )

    print(
        "Direction-cell spikes:",
        int(
            np.sum(
                direction_spikes
            )
        )
    )

    goal_evidence = (
        calculate_goal_evidence(
            observed_x,
            observed_y,
            hole_x,
            hole_y
        )
    )

    learned_goal = get_learned_goal(
        observed_x,
        observed_y,
        hole_x,
        hole_y,
        goal_evidence
    )

    print()

    print(
        "Goal evidence after observation:"
    )

    for i in range(
        NUM_HOLES
    ):

        print(
            "Hole",
            i,
            ":",
            round(
                goal_evidence[i],
                3
            )
        )

    print()

    print(
        "=========================================="
    )

    print(
        "OBSERVATION FINISHED"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "The demonstrator is no longer observed."
    )

    print(
        "The observer now searches independently."
    )

    print(
        "Observer starting position: centre"
    )

    print(
        "Observer will continue until the goal is found."
    )

    print()

    (
        observer_x,
        observer_y,
        search_time,
        visited_holes
    ) = run_observer(
        hole_x,
        hole_y,
        goal_hole,
        learned_goal
    )

    print()

    print(
        "=========================================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Observation time:",
        OBSERVATION_TIME,
        "seconds"
    )

    print(
        "Actual goal:",
        goal_hole
    )

    print(
        "Holes visited:"
    )

    print(
        visited_holes
    )

    print()

    print(
        "Number of different holes visited:",
        len(
            set(
                visited_holes
            )
        )
    )

    print()

    print(
        "Goal reached!"
    )

    print(
        "Observer search time:",
        round(
            search_time,
            2
        ),
        "seconds"
    )

    print(
        "Total time including observation:",
        round(
            OBSERVATION_TIME
            + search_time,
            2
        ),
        "seconds"
    )

    print()

    observer_distance = np.sum(
        np.sqrt(
            np.diff(
                observer_x
            ) ** 2
            +
            np.diff(
                observer_y
            ) ** 2
        )
    )

    print(
        "Observer distance travelled:",
        round(
            observer_distance,
            2
        ),
        "cm"
    )

    print()

    results.append(
        {
            "observation_time": OBSERVATION_TIME,
            "search_time": search_time,
            "total_time": (
                OBSERVATION_TIME
                + search_time
            ),
            "visited_holes": visited_holes,
            "observer_x": observer_x,
            "observer_y": observer_y
        }
    )

    plot_demonstrator(
        hole_x,
        hole_y,
        goal_hole,
        start_x,
        start_y,
        observed_x,
        observed_y
    )

    plot_spatial_activity(
        hole_x,
        hole_y,
        goal_hole,
        firing_positions
    )

    plot_observer(
        hole_x,
        hole_y,
        goal_hole,
        observer_x,
        observer_y,
        visited_holes,
        search_time
    )


print()
print(
    "=========================================="
)
print(
    "COMBINED RESULTS"
)
print(
    "=========================================="
)
print()

for result in results:

    print(
        "Observation:",
        result["observation_time"],
        "seconds"
    )

    print(
        "Observer search time:",
        round(
            result["search_time"],
            2
        ),
        "seconds"
    )

    print(
        "Total time including observation:",
        round(
            result["total_time"],
            2
        ),
        "seconds"
    )

    print(
        "Different holes visited:",
        len(
            set(
                result["visited_holes"]
            )
        )
    )

    print()
