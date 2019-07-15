import numpy as np
from agent import Agent, Option
from environment import Environment
from sim_funs import find_state


class Simulation():
    def __init__(self, agent_params, env_params, option_params, sim_params):
        self.mode = sim_params["task_mode"]
        self.num_trials = sim_params["num_trials"]
        self.regimes = sim_params["regime"]
        self.active_regime = self.regimes[0]

        self.agent = Agent(
            agent_params["alpha"],
            agent_params["gamma"],
            agent_params["action_lbls"],
            agent_params["policy"]
        )
        self.agent.init_primitive_actions(self.mode)

        num_options = len(option_params["label"])
        for n in range(num_options):
            o = {}
            for key, value in option_params.items():
                o[key] = value[n]
            option = Option(o)
            self.agent.add_option(option)

        self.env = Environment(
            env_params["states"],
            env_params["state_labels"]
        )
        self.env.init_pR()

        if self.mode is "hierarchical":
            self.SG_sides = ["L", "R"]
        elif self.mode is "flat":
            self.SG_sides = ["L", "L"]

        self.G_sides = ["L", "R"]

        self.num_steps = []
        self.mu_steps = []

    def setup_trial(self):
        self.agent.termination_reached = False

        self.G_side = self.SG_sides[self.n_trial % 2]
        if self.active_regime is "repeat":
            self.SG_side = self.G_side
        elif self.active_regime is "alternate":
            self.SG_side = [
                side for side in self.G_sides if side != self.G_side
            ][0]
        SG_i = self.SG_sides.index(self.SG_side)

        self.agent.reset(self.SG_side, self.env)

        self.env.init_pR()
        self.env.place_r(self.G_side, self.SG_side, SG_i)

        self.t = 0

    def record_trial(self):
        self.num_steps.append(self.t)
        self.mu_steps.append(np.round(np.mean(self.num_steps)))

    def switch_regime(self):
        self.active_regime = self.regimes[1]
        print("---------------------------- REGIME SWITCH --------------------"
              "--------")

    def norm_Q(self):
        self.agent.Q[:, :] = self.agent.Q[:, :] / sum(sum(self.agent.Q[:, :]))

    def summarise_step(self):
        print("t_{0}, S: {1}, S': {2}, A: {3}, "
              "r: {4}, SG_v: {5}".format(
                  self.t,
                  self.env.states[find_state(self.agent.prev_state, self.env)],
                  self.env.states[find_state(self.agent.state, self.env)],
                  self.agent.action_lbls[self.agent.a_i],
                  self.agent.r,
                  self.agent.SG_visited)
              )
        # print(np.round(self.agent.Q, 3))

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
            print("-----------------------------------------------------------"
                  "------------\n"
                  "        trial num = {0}\n"
                  " mean steps taken = {1}\n"
                  "         Q(B0, :) = {2}\n"
                  "        Q(SGL, :) = {3}\n"
                  "        Q(SGR, :) = {4}\n"
                  "         Q(B1, :) = {5}\n"
                  "-----------------------------------------------------------"
                  "------------".format(
                      self.n_trial, self.mu_steps[-1],
                      np.round(self.agent.Q[:, 0], 3),
                      np.round(self.agent.Q[:, 1], 3),
                      np.round(self.agent.Q[:, 2], 3),
                      np.round(self.agent.Q[:, 4], 3))
                  )
        elif self.mode is "hierarchical":
            print("-----------------------------------------------------------"
                  "------------\n"
                  "        trial num = {0}\n"
                  " mean steps taken = {1}\n"
                  "        Q(B0L, :) = {2}\n"
                  "        Q(B0R, :) = {3}\n"
                  "        Q(SGL, :) = {4}\n"
                  "        Q(SGR, :) = {5}\n"
                  "         Q(B1, :) = {6}\n"
                  "-----------------------------------------------------------"
                  "------------".format(
                      self.n_trial, self.mu_steps[-1],
                      np.round(self.agent.Q[:, 0], 2),
                      np.round(self.agent.Q[:, 1], 2),
                      np.round(self.agent.Q[:, 2], 2),
                      np.round(self.agent.Q[:, 3], 2),
                      np.round(self.agent.Q[:, 5], 2))
                  )
