import numpy as np


class Environment():
    def __init__(self, states):
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
