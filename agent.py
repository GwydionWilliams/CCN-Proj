import numpy as np


class Agent():
    def __init__(self, actions, alpha, gamma, policy, epsilon=.1):
        self.Q = None
        self.r = 0

        self.prev_state = None
        self.state = None
        self.action = None
        self.termination_reached = False

        self.actions = actions

        self.num_actions = len(actions)

        self.alpha = alpha
        self.gamma = gamma

        self.policy = policy
        if self.policy == "e-greedy":
            self.epsilon = epsilon
        elif self.policy == "greedy":
            self.epsilon = 1

    def init_Q(self, env, allowable_movements):
        self.Q = np.zeros((self.num_actions, env.num_states))

        for s in range(env.num_states):
            possible_movements = allowable_movements[s]
            self.Q[possible_movements, s] = 1  # \
            # np.round(1/len(possible_movements), 3)

        print(self.Q)

    def select_action(self):
        Q_choices = self.Q[:, self.state]

        if np.random.random() < self.epsilon:
            self.action = np.random.choice(np.where(Q_choices > 0)[0])
        else:
            choice = np.where(Q_choices == max(Q_choices))[0]
            if len(choice) > 1:
                self.action = np.random.choice(choice)
            else:
                self.action = choice[0]

    def move(self, env):
        self.prev_state = self.state

        self.state = np.where(
            env.T[self.actions[self.action]][self.prev_state, :] == 1
        )[0][0]

        if env.states[self.state] == env.SG:
            self.SG_visited = True

    def collect_reward(self, env):
        if self.SG_visited:
            self.r = env.pR[self.state]
        else:
            self.r = 0

        if self.r == 1:
            self.termination_reached = True

    def update_Q(self):
        self.Q[self.action, self.prev_state] = \
            self.Q[self.action, self.prev_state] + \
            self.alpha*(self.r + self.gamma *
                        np.max(self.Q[:, self.state]) -
                        self.Q[self.action, self.prev_state])

    def reset(self, SG_side):
        self.r = 0

        if SG_side is "L":
            self.state = 0
        else:
            self.state = 1

        self.SG_visited = False
