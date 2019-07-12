import numpy as np
from agent import Agent
from environment import Environment

actions = ["NE", "SE", "SW", "NW"]
num_actions = len(actions)
alpha = .5
gamma = .5
policy = "e-greedy"

agent = Agent(num_actions, actions, alpha, gamma, policy)

state_labels = ["B0", "SGL", "SGR", "DL", "B1", "DR", "GL", "GR"]
num_states = len(state_labels)
states = np.arange(num_states)

env = Environment(num_states, states, state_labels)

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
agent.init_Q(env, allowable_movements)

allowable_transitions = {
    "NE": [[0, 2], [1, 4], [2, 5], [4, 7]],
    "SE": [[1, 0], [3, 1], [4, 2], [6, 4]],
    "SW": [[2, 0], [4, 1], [5, 2], [7, 4]],
    "NW": [[0, 1], [1, 3], [2, 4], [4, 6]]
}
env.init_T(allowable_transitions)
env.init_pR()

num_trials = int(1e3)

for n in range(num_trials):
    print("------------------------- NEW TRIAL -------------------------")

    termination_reached = False
    agent.reset()

    while agent.r != 1:
        agent.select_action()
        agent.move(env)
        agent.collect_reward(env)
        agent.update_Q()

        print("S: {0}, S': {1}, A: {2}, r: {3}, Q(S, A): {4}".format(
            env.state_labels[agent.prev_state], env.state_labels[agent.state],
            agent.actions[agent.action],
            agent.r, np.round(agent.Q[agent.action, agent.prev_state], 2))
        )

    print("------------------------ TRIAL ENDED ------------------------")
    for s_i in [0, 1, 2, 4, 5]:
        s = env.states[s_i]
        a_max = np.where(
            agent.Q[:, s] == max(agent.Q[:, s])
        )[0]
        p_a = np.round(max(agent.Q[:, s]), 3)

        print("Most likely action(s) from {0}: {1}, with Q={2}".format(
            env.state_labels[s], a_max, p_a
        ))
