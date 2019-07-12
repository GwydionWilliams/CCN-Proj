import numpy as np


class Environment():
    def __init__(self, states):
        self.T = None
        self.states = states
        self.num_states = len(states)
        self.state_i = np.arange(self.num_states)

    def init_T(self, allowable_transitions):
        self.T = {}

        for a in allowable_transitions.keys():
            num_transitions = len(allowable_transitions[a])
            self.T[a] = np.zeros(
                (self.num_states, self.num_states)
            )

            for t in range(num_transitions):
                idx = allowable_transitions[a][t]
                self.T[a][idx[0], idx[1]] = 1

    def init_pR(self):
        self.pR = np.zeros(self.num_states)

    def place_r(self, SG_side):
        self.pR[self.states.index("G" + SG_side)] = 1
        self.SG = "SG" + SG_side


def env_builder(mode):
    if mode is "hierarchical":
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

    return allowable_transitions, allowable_movements
