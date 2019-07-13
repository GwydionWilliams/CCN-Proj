import numpy as np
from agent import Agent, Option
from environment import Environment


class Simulation():
    def __init__(self, agent_params, env_params, num_trials, mode):
        self.agent = Agent(
            agent_params["actions"],
            agent_params["alpha"],
            agent_params["gamma"],
            agent_params["policy"]
        )

        print(self.agent.Q)

        self.env = Environment(
            env_params["states"]
        )

        self.agent.init_Q(self.env, agent_params["allowable_movements"])
        self.env.init_T(env_params["allowable_transitions"])
        self.env.init_pR()

        self.num_trials = num_trials

        self.mode = mode
        if mode is "hierarchical":
            self.SG_sides = ["L", "R"]
        elif mode is "flat":
            self.SG_sides = ["L", "L"]

        self.num_steps = []
        self.mu_steps = []

        s_initiation = 0
        s_termination = 4
        pi = [[3, 0], [0, 1]]
        Option(pi, s_termination, s_initiation, self.agent)

    def setup_trial(self):

        self.agent.termination_reached = False

        self.SG_side = self.SG_sides[self.n_trial % 2]

        self.agent.reset(self.SG_side)

        self.env.init_pR()
        self.env.place_r(self.SG_side)

        self.t = 0

    def record_trial(self):
        self.num_steps.append(self.t)
        self.mu_steps.append(np.round(np.mean(self.num_steps)))

    def norm_Q(self):
        self.agent.Q[:, :] = self.agent.Q[:, :] / sum(sum(self.agent.Q[:, :]))

    def summarise_step(self):
        print("t_{0}, S: {1}, S': {2}, A: {3}, "
              "r: {4}, Q(S, A): {5}, SG_v: {6}".format(
                  self.t,
                  self.env.states[self.agent.prev_state],
                  self.env.states[self.agent.state],
                  self.agent.actions[self.agent.action],
                  self.agent.r,
                  np.round(
                      self.agent.Q[self.agent.action, self.agent.prev_state], 2
                  ),
                  self.agent.SG_visited)
              )

    def summarise_trial(self):

        if self.mode is "hierarchical":
            print("steps taken this trial: {0}, mean steps taken = {1},\n"
                  "Q(B0L, :) = {2}, \nQ(B0R, :) = {3}, \nQ(B1, :) = {4}".
                  format(
                      self.t,
                      self.mu_steps[-1],
                      np.round(self.agent.Q[:, 0], 3),
                      np.round(self.agent.Q[:, 1], 3),
                      np.round(self.agent.Q[:, 5], 3),
                  ))

        elif self.mode is "flat":
            print("steps taken this trial: {0}, mean steps taken = {1},\n"
                  "Q(B0, :) = {2}, \nQ(B1, :) = {3}".
                  format(
                      self.t,
                      self.mu_steps[-1],
                      np.round(self.agent.Q[:, 0], 3),
                      np.round(self.agent.Q[:, 4], 3),
                  ))

    def summarise_chunk(self):
        if self.mode is "flat":
            print("---------------------------------------------\n"
                  "        trial num = {0}\n"
                  " mean steps taken = {1}\n"
                  "         Q(B0, :) = {2}\n"
                  "        Q(SGL, :) = {3}\n"
                  "        Q(SGR, :) = {4}\n"
                  "         Q(B1, :) = {5}\n"
                  "---------------------------------------------".format(
                      self.n_trial, self.mu_steps[-1],
                      np.round(self.agent.Q[:, 0], 3),
                      np.round(self.agent.Q[:, 1], 3),
                      np.round(self.agent.Q[:, 2], 3),
                      np.round(self.agent.Q[:, 4], 3))
                  )
        elif self.mode is "hierarchical":
            print("---------------------------------------------\n"
                  "        trial num = {0}\n"
                  " mean steps taken = {1}\n"
                  "        Q(B0L, :) = {2}\n"
                  "        Q(B0R, :) = {3}\n"
                  "        Q(SGL, :) = {4}\n"
                  "        Q(SGR, :) = {5}\n"
                  "         Q(B1, :) = {6}\n"
                  "---------------------------------------------".format(
                      self.n_trial, self.mu_steps[-1],
                      np.round(self.agent.Q[:, 0], 3),
                      np.round(self.agent.Q[:, 1], 3),
                      np.round(self.agent.Q[:, 2], 3),
                      np.round(self.agent.Q[:, 3], 3),
                      np.round(self.agent.Q[:, 5], 3))
                  )
