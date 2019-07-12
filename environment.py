import numpy as np


class Environment():
    def __init__(self, num_states, states, state_labels):
        self.num_states = num_states
        self.states = states
        self.state_labels = state_labels

    def init_T(self, allowable_transitions):
        self.T = {}
        allowable_transitions = {
            "NE": [[0, 2], [1, 4], [2, 5], [4, 7]],
            "SE": [[1, 0], [3, 1], [4, 2], [6, 4]],
            "SW": [[2, 0], [4, 1], [5, 2], [7, 4]],
            "NW": [[0, 1], [1, 3], [2, 4], [4, 6]]
        }

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
        self.pR[-1] = 1
