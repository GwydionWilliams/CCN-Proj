from simulation import Simulation
from sim_funs import build_env, write_data

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
mode = "flat"
num_trials = int(1e5)

#    ii.  AGENT & ENVIRONMENT
actions = ["NE", "SE", "SW", "NW"]
alpha = .01
gamma = .5
policy = "e-greedy"
epsilon = .05

states, state_labels, allowable_movements = build_env(mode)

agent_params = {
    "actions": actions,
    "alpha": alpha,
    "gamma": gamma,
    "policy": policy,
    "epsilon": epsilon,
    "allowable_movements": allowable_movements
}

env_params = {
    "states": states,
    "state_labels": state_labels
}

#    iii. DATA
data_dir = "./data/"
file_name = "10_MFH-flatEnv"

#    iv. CONTROLLER
sim = Simulation(agent_params, env_params, num_trials, mode)


# -----------------------------------------------------------------------------
# 2. RUN SIMULATION -----------------------------------------------------------
for sim.n_trial in range(sim.num_trials):

    sim.setup_trial()
    sim.t = 0

    while sim.agent.termination_reached is not True:

        sim.agent.select_action(sim.env)
        sim.agent.move(sim.env)
        sim.agent.collect_reward(sim.env, sim.mode)
        sim.agent.update_Q(sim.env)

        # sim.summarise_step()

        sim.t += 1

    sim.record_trial()

    if sim.n_trial != 0:
        if ((sim.n_trial % (sim.num_trials / 10)) == 0) or \
           (sim.n_trial == (sim.num_trials-1)):
            sim.summarise_chunk()

# -----------------------------------------------------------------------------
# 3. SAVE RESULTS -------------------------------------------------------------
write_data(sim, data_dir, file_name)
