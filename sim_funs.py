import csv


def build_env(mode):
    if mode is "hierarchical":
        states = ["B0L", "B0R", "SGL", "SGR", "DL", "B1", "DR", "GL", "GR"]
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
        allowable_transitions = {
            "NE": [[0, 3], [1, 3], [2, 5], [3, 6], [5, 8]],
            "SE": [[2, 0], [2, 1], [4, 2], [5, 3], [7, 5]],
            "SW": [[3, 0], [3, 1], [5, 2], [6, 3], [8, 5]],
            "NW": [[0, 2], [1, 2], [2, 4], [3, 5], [5, 7]]
        }

    elif mode is "flat":
        states = ["B0", "SGL", "SGR", "DL", "B1", "DR", "GL", "GR"]
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
        allowable_transitions = {
            "NE": [[0, 2], [1, 4], [2, 5], [4, 7]],
            "SE": [[1, 0], [3, 1], [4, 2], [6, 4]],
            "SW": [[2, 0], [4, 1], [5, 2], [7, 4]],
            "NW": [[0, 1], [1, 3], [2, 4], [4, 6]]
        }

    return states, allowable_movements, allowable_transitions


def write_data(sim, dir_name, file_name):
    with open(dir_name + "/" + file_name + ".csv", 'w') as writeFile:
        writer = csv.writer(writeFile)

        output = [sim.num_steps, sim.mu_steps]
        output = zip(*output)

        writer.writerows(output)
