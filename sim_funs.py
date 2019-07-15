import csv
import numpy as np


def build_env(mode):
    if mode is "hierarchical":
        states = [[0, 0, 0], [0, 0, 1],
                  [-1, 1, 0], [1, 1, 0], [-2, 2, 0], [0, 2, 0], [2, 2, 0],
                  [-1, 3, 0], [1, 3, 0]]
        state_labels = ["B0L", "B0R",
                        "SGL", "SGR", "DL", "B1", "DR",
                        "GL", "GR"]

    elif mode is "flat":
        states = [[0, 0, 0], [-1, 1, 0], [1, 1, 0],
                  [-2, 2, 0], [0, 2, 0], [2, 2, 0],
                  [-1, 3, 0], [1, 3, 0]]
        state_labels = ["B0", "SGL", "SGR",
                        "DL", "B1", "DR",
                        "GL", "GR"]

    return states, state_labels


def define_primitive_actions(action_lbls, mode):
    s_initiation = {
        action_lbls[0]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        action_lbls[1]: [0, 0, 1, 0, 1, 0, 0, 1, 0],
        action_lbls[2]: [0, 0, 0, 1, 0, 1, 1, 0, 1],
        action_lbls[3]: [1, 1, 1, 1, 0, 1, 0, 0, 0]
    }

    s_termination = {
        action_lbls[0]: [0, 0, 0, 1, 0, 1, 1, 0, 1],
        action_lbls[1]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        action_lbls[2]: [1, 1, 1, 1, 0, 1, 0, 0, 0],
        action_lbls[3]: [0, 0, 1, 0, 1, 1, 0, 1, 0]
    }

    if mode is "flat":
        for a in action_lbls:
            s_initiation[a] = s_initiation[a][1:]
            s_termination[a] = s_termination[a][1:]

    num_actions = len(action_lbls)
    num_states = len(s_initiation[action_lbls[0]])

    pi = {}
    for i, a in enumerate(s_initiation.keys()):
        pi[a] = np.zeros((num_actions, num_states))
        for s in range(num_states):
            if s_initiation[a][s] == 1:
                pi[a][i, s] = 1

    return s_initiation, s_termination, pi


def define_options(agent_class, task_mode):
    if agent_class is "flat":
        labels = s_init = s_term = pi = []
    elif agent_class is "hierarchical":
        labels = ["B0_B1_L", "B0_B1_R",
                  "B0_GL_R", "B0_GR_R",
                  "B0_GL_A", "B0_GR_A"]

        s_init = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
        ]

        s_term = [
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        ]

        pi = [
            np.array([
                0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((4, 9)),
            np.array([
                1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0,
            ]).reshape((4, 9)),
            np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
            np.array([
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
            np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
            np.array([
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]).reshape((6, 9)),
        ]

        if task_mode is "flat":
            for o in range(len(labels)):
                s_init[o] = s_init[0][1:]
                s_term[o] = s_term[0][1:]
                pi[o] = pi[o][:, 1:]

    return labels, s_init, s_term, pi


def write_data(sim, dir_name, file_name):
    with open(dir_name + "/" + file_name + ".csv", 'w') as writeFile:
        writer = csv.writer(writeFile)

        output = [sim.num_steps, sim.mu_steps]
        output = zip(*output)

        writer.writerows(output)


def find_state(state, env, value="index"):
    if value is "index":
        return env.states.index(state)
    elif value is "label":
        return env.state_lbls[env.states.index(state)]
