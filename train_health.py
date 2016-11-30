from RestrictedEnvironment import Level
from run import run_experiment
from config import *


# resultDir = "/media/arpit/datadisk/private/10701/project/results/exp14/"
resultDir = "./../"

import matplotlib.pyplot as plt

train_param = {
            "snapshot_episodes": 100,
            "episodes": 10,
            "steps_per_episode": 200, # 4300 for deathmatch, 300 for health gathering
            "average_over_num_episodes": 50,
            "start_learning_after": 0,
            "algorithm": Algorithm.DDQN,
            "discount": 0.99,
            "max_memory": 5000,
            "prioritized_experience": False,
            "exploration_policy": ExplorationPolicy.E_GREEDY,
            "learning_rate": 2.5e-4,
            "level": Level.HEALTH,
            "combine_actions": False,
            "temperature": 10,
            "batch_size": 1,
            "history_length": 4,
            "snapshot": '',
            "snapshot_itr_num": 0,
            "mode": Mode.TRAIN,
            "skipped_frames": 4,
            "target_update_freq": 1000,
            "steps_between_train": 1,
            "epsilon_start": 0.5,
            "epsilon_end": 0.01,
            "epsilon_annealing_steps": 3e4,
            "architecture": Architecture.DIRECT,
            "max_action_sequence_length": 1,
            "log_dir":resultDir,
            "log_after_episodes":10,
            "visible":False,
            "save_results_dir": resultDir,
            "visible":False,
            "save_ERM":"",
            "load_ERM":False,
            "ERM":None
        }



# run agent
returns, Qs, memory = run_experiment(train_param)

# plot results
plt.figure(1)
plt.plot(range(len(returns)), returns)
plt.savefig('av_return_10k.png')
plt.xlabel("episode")
plt.ylabel("average return")
plt.title("Average Return")

plt.figure(2)
plt.plot(range(len(Qs)), Qs)
plt.savefig('av_q_10k.png')
plt.xlabel("episode")
plt.ylabel("mean Q value")
plt.title("Mean Q Value")

plt.show()