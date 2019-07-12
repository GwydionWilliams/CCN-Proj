import numpy as np
from agentControllers import *

actions = ["NE", "SE", "SW", "NW"]
num_actions = len(actions)

num_states = 8
states = np.arange(num_states)
state_labels = ["B0", "SGL", "SGR", "DL", "B1", "DR", "GL", "GR"]

Q = np.zeros((num_actions, num_states))
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
for s in range(num_states):
    possible_movements = allowable_movements[s]
    Q[possible_movements, s] = np.round(1/len(possible_movements), 3)

pT = {}
allowable_transitions = {
    "NE": [[0, 2], [1, 4], [2, 5], [4, 7]],
    "SE": [[1, 0], [3, 1], [4, 2], [6, 4]],
    "SW": [[2, 0], [4, 1], [5, 2], [7, 4]],
    "NW": [[0, 1], [1, 3], [2, 4], [4, 6]]
}

for action in actions:
    num_transitions = len(allowable_transitions[action])
    pT[action] = np.zeros((num_states, num_states))

    for t in range(num_transitions):
        idx = allowable_transitions[action][t]

        pT[action][idx[0], idx[1]] = 1

pR = np.zeros(num_states)
pR[-1] = 1

num_trials = int(1000)

alpha = .1
gamma = 1

for n in range(num_trials):
    termination_reached = False
    s_current = 0

    while not termination_reached:
        choice = np.where(Q[:, s_current] == max(Q[:, s_current]))[0]
        if len(choice) > 1:
            a = np.random.choice(choice)
        else:
            a = choice[0]

        s_next = np.where(pT[actions[a]][s_current, :] == 1)[0][0]
        r = pR[s_next]

        Q[a, s_current] = \
            Q[a, s_current] + \
            alpha*(r + gamma*np.max(Q[a, s_next]) - Q[a, s_current])

        print("Current state: {0}, reward: {1}, Q(S, A): {2}, a: {3}".format(
            state_labels[s_current], r,
            np.round(Q[a, s_current], 2), actions[a])
        )

        s_current = s_next

        if r == 1:
            termination_reached = True
            print("----------------------------------------------------------")
