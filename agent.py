import numpy as np
from sim_funs import find_state


class Agent():
    def __init__(self, actions, alpha, gamma, policy, epsilon=.1):
        self.Q = None
        self.pi_active = None

        self.r = 0

        self.prev_state = None
        self.state = None
        self.action = None
        self.termination_reached = False

        self.actions = actions

        self.num_actions = len(actions)

        self.alpha = alpha
        self.gamma = gamma

        self.selection_policy = policy
        if self.selection_policy == "e-greedy":
            self.epsilon = epsilon
        elif self.selection_policy == "greedy":
            self.epsilon = 0

    def init_Q(self, env, allowable_movements):
        self.Q = np.zeros((self.num_actions, env.num_states))

        for s in range(env.num_states):
            possible_movements = allowable_movements[s]
            self.Q[possible_movements, s] = 1  # \
            # np.round(1/len(possible_movements), 3)

        self.pi_active = self.Q

    def add_option(self, option):
        Q_o = np.zeros(self.Q.shape[1])
        for s in option.s_initiation:
            Q_o[option.s_initiation] = 1
        self.Q = np.vstack((self.Q, Q_o))

    def select_action(self, env):
        s_i = find_state(self.state, env)
        choices = self.Q[:, s_i]

        if np.random.random() < self.epsilon:
            choice = np.random.choice(np.where(choices > 0)[0])
        else:
            choice = np.where(choices == max(choices))[0]
            if len(choice) > 1:
                choice = np.random.choice(choice)
            else:
                choice = choice[0]
        self.action = choice

        # if choice < 4:
        #     self.action = choice
        # else:
        #     self.pi_active =

        # if self.action > 4:

    def move(self, env):
        self.prev_state = self.state[:]

        x_shift = 1
        if self.actions[self.action][1] == "W":
            x_shift *= -1

        y_shift = 1
        if self.actions[self.action][0] == "S":
            y_shift *= -1

        self.state[0] += x_shift
        self.state[1] += y_shift

        if find_state(self.state, env, value="label") == env.SG:
            self.SG_visited = True

    def collect_reward(self, env, mode):
        s_i = find_state(self.state, env)

        if mode == "hierarchical":
            if self.SG_visited:
                self.r = env.pR[s_i]
            else:
                self.r = 0
        elif mode == "flat":
            self.r = env.pR[s_i]

        if self.r == 1:
            self.termination_reached = True

    def update_Q(self, env):
        s_prev = find_state(self.prev_state, env)
        s_curr = find_state(self.state, env)

        self.Q[self.action, s_prev] = \
            self.Q[self.action, s_prev] + \
            self.alpha*(self.r + self.gamma *
                        np.max(self.Q[:, s_curr]) -
                        self.Q[self.action, s_prev])

    def reset(self, SG_side):
        self.r = 0

        if SG_side is "L":
            self.state = [0, 0, 0]
        else:
            self.state = [0, 0, 1]

        self.SG_visited = False


class Option():
    def __init__(self, pi, s_termination, s_initiation, agent):
        self.s_termination = s_termination
        self.s_initiation = s_initiation

        self.pi = np.zeros(agent.Q.shape)
        for a in pi:
            self.pi[a[0], a[1]] = 1
