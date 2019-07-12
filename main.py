from simulation import Simulation
from environment import env_builder

mode = "hierarchical"
alpha = .05
gamma = .5
policy = "e-greedy"
epsilon = .1
allowable_movements, allowable_transitions = env_builder(mode)

agent_params = {
    "actions": ["NE", "SE", "SW", "NW"],
    "alpha": alpha,
    "gamma": gamma,
    "policy": policy,
    "epsilon": epsilon,
    "allowable_movements": allowable_movements
}

env_params = {
    "states": ["B0L", "B0R", "SGL", "SGR", "DL", "B1", "DR", "GL", "GR"],
    "allowable_transitions": allowable_transitions
}

num_trials = int(1e5)

sim = Simulation(agent_params, env_params, num_trials, mode)

for sim.n_trial in range(sim.num_trials):

    sim.setup_trial()
    sim.t = 0

    while sim.agent.termination_reached is not True:

        sim.agent.select_action()
        sim.agent.move(sim.env)
        sim.agent.collect_reward(sim.env)
        sim.agent.update_Q()

        # sim.summarise_step()

        sim.t += 1

    sim.norm_Q()

    sim.summarise_trial()
