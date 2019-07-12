import numpy as np
from agent import Agent
from environment import Environment


class Simulation():
    def __init__(self,
                 agent_params, env_params, num_trials,
                 reward_mode="hierarchical"):
        self.agent = Agent(
            agent_params["actions"],
            agent_params["alpha"],
            agent_params["gamma"],
            agent_params["policy"]
        )

        self.env = Environment(
            env_params["states"]
        )

        self.agent.init_Q(self.env, agent_params["allowable_movements"])
        self.env.init_T(env_params["allowable_transitions"])

        self.env.init_pR()

        self.num_trials = num_trials

        if reward_mode is "hierarchical":
            self.SG_sides = ["L", "R"]
        elif reward_mode is "simple":
            self.SG_sides = ["L", "L"]

        self.n_steps = []
        self.mu_steps = []

    def setup_trial(self):
        print("------------------------- NEW TRIAL -------------------------")

        self.agent.termination_reached = False

        self.SG_side = self.SG_sides[self.n_trial % 2]

        self.agent.reset(self.SG_side)

        self.env.init_pR()
        self.env.place_r(self.SG_side)

        self.t = 0

    def summarise_step(self):
        print("t_{0}, S: {1}, S': {2}, A: {3}," +
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

    def norm_Q(self):
        for s in self.env.state_i:
            self.agent.Q[:, s] = self.agent.Q[:, s] / sum(self.agent.Q[:, s])

    def summarise_trial(self):
        self.n_steps.append(self.t)
        self.mu_steps.append(np.round(np.mean(self.n_steps)))

        print("steps taken this trial: {0}, mean steps taken = {1},\n"
              "Q(B0L, :) = {3}, \nQ(B0R, :) = {3}, \nQ(B1, :) = {4}".format(
                  self.t,
                  self.mu_steps[-1],
                  np.round(self.agent.Q[:, 0], 3),
                  np.round(self.agent.Q[:, 1], 3),
                  np.round(self.agent.Q[:, 5], 3),
              ))
