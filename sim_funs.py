import csv


def build_env(mode):
    if mode is "hierarchical":
        states = [[0, 0, 0], [0, 0, 1],
                  [-1, 1, 0], [1, 1, 0], [-2, 2, 0], [0, 2, 0], [2, 2, 0],
                  [-1, 3, 0], [1, 3, 0]]
        state_labels = ["B0L", "B0R",
                        "SGL", "SGR", "DL", "B1", "DR",
                        "GL", "GR"]
        allowable_movements = {
            0: [0, 3],
            1: [0, 3],
            2: [0, 1, 3],
            3: [0, 2, 3],
            4: [1],
            5: [0, 1, 2, 3],
            6: [2],
            7: [1],
            8: [2],
        }

    elif mode is "flat":
        states = [[0, 0, 0], [-1, 1, 0], [1, 1, 0],
                  [-2, 2, 0], [0, 2, 0], [2, 2, 0],
                  [-1, 3, 0], [1, 3, 0]]
        state_labels = ["B0", "SGL", "SGR",
                        "DL", "B1", "DR",
                        "GL", "GR"]
        allowable_movements = {
            0: [0, 3],
            1: [0, 1, 3],
            2: [0, 2, 3],
            3: [1],
            4: [0, 1, 2, 3],
            5: [2],
            6: [1],
            7: [2],
        }

    return states, state_labels, allowable_movements


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
        return env.state_labels[env.states.index(state)]
