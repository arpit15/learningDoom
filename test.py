from RestrictedEnvironment import Level
from run import run_experiment
from config import *

result_dir = "/media/arpit/datadisk/private/10701/project/results/"
train_param = {
            "snapshot_episodes": 100,
            "episodes": 5,
            "steps_per_episode": 300, # 4300 for deathmatch, 300 for health gathering
            "average_over_num_episodes": 50,
            "start_learning_after": 20,
            "algorithm": Algorithm.DDQN,
            "discount": 0.99,
            "max_memory": 10000,
            "prioritized_experience": False,
            "exploration_policy": ExplorationPolicy.E_GREEDY,
            "learning_rate": 2.5e-4,
            "level": Level.DEFEND,        # change to the desired env
            "combine_actions": True,      # False only for Deathmatch
            "temperature": 10,
            "batch_size": 10,
            "history_length": 4,
            "snapshot":'exp6.h5',         # h5 model file name
            "snapshot_itr_num": 10000,
            "mode": Mode.DISPLAY,
            "skipped_frames": 4,
            "target_update_freq": 1000,
            "steps_between_train": 1,
            "epsilon_start": 0.5,
            "epsilon_end": 0.01,
            "epsilon_annealing_steps": 3e4,
            "architecture": Architecture.DIRECT,
            "max_action_sequence_length": 1,
            "save_results_dir": result_dir,
            "visible":True,
            "save_ERM":"./"
        }




# run agent
returns, Qs = run_experiment(train_param)

# plot results
import matplotlib.pyplot as plt

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