from simulation import Simulation

reward_mode = "hierarchical"

if reward_mode is "hierarchical":
    agent_params = {
        "actions": ["NE", "SE", "SW", "NW"],
        "alpha": .1,
        "gamma": .5,
        "policy": "e-greedy",
        "allowable_movements": {
            0: [0, 3],
            1: [0, 3],
            2: [0, 1, 3],
            3: [0, 2, 3],
            4: [1],
            5: [0, 1, 2, 3],
            6: [2],
            7: [1],
            8: [2],
        }
    }

    env_params = {
        "states": ["B0L", "B0R", "SGL", "SGR", "DL", "B1", "DR", "GL", "GR"],
        "allowable_transitions": {
            "NE": [[0, 3], [1, 3], [2, 5], [3, 6], [5, 8]],
            "SE": [[2, 0], [2, 1], [4, 2], [5, 3], [7, 5]],
            "SW": [[3, 0], [3, 1], [5, 2], [6, 3], [8, 5]],
            "NW": [[0, 2], [1, 2], [2, 4], [3, 5], [5, 7]]
        }
    }

elif reward_mode is "simple":


num_trials = int(1e5)

sim = Simulation(agent_params, env_params, num_trials)

for sim.n_trial in range(sim.num_trials):

    sim.setup_trial()

    while sim.agent.termination_reached is not True:

        sim.agent.select_action()
        sim.agent.move(sim.env)
        sim.agent.collect_reward(sim.env)
        sim.agent.update_Q()

        # sim.summarise_step()

        sim.t += 1

    sim.norm_Q()

    sim.summarise_trial()
