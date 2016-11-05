from Environment import Level
from main_final import run_experiment, ExplorationPolicy, Algorithm, Mode, Architecture
import matplotlib.pyplot as plt


# train_param = {
#             "snapshot_episodes": 100,
#             "episodes": 10000,
#             "steps_per_episode": 400, # 4300 for deathmatch, 300 for health gathering
#             "average_over_num_episodes": 50,
#             "start_learning_after": 10,
#             "algorithm": Algorithm.DQN,
#             "discount": 0.99,
#             "max_memory": 1000,
#             "prioritized_experience": False,
#             "exploration_policy": ExplorationPolicy.E_GREEDY,
#             "learning_rate": 2.5e-4,
#             "level": Level.BASIC,
#             "combine_actions": False,
#             "temperature": 10,
#             "batch_size": 10,
#             "history_length": 4,
#             "snapshot": '/media/arpit/datadisk/private/10701/project/results/exp5/models/model_1000.h5',
#             "snapshot_itr_num": 1000,
#             "mode": Mode.TRAIN,
#             "skipped_frames": 4,
#             "target_update_freq": 1000,
#             "steps_between_train": 1,
#             "epsilon_start": 0.5,
#             "epsilon_end": 0.01,
#             "epsilon_annealing_steps": 3e4,
#             "architecture": Architecture.DIRECT,
#             "max_action_sequence_length": 5,
            
#         }

train_param = {
            "snapshot_episodes": 1000,
            "episodes": 10000,
            "steps_per_episode": 40, # 4300 for deathmatch, 300 for health gathering
            "average_over_num_episodes": 50,
            "start_learning_after": 20,
            "algorithm": Algorithm.DDQN,
            "discount": 0.99,
            "max_memory": 5000,
            "prioritized_experience": False,
            "exploration_policy": ExplorationPolicy.E_GREEDY,
            "learning_rate": 2.5e-4,
            "level": Level.BASIC,
            "combine_actions": True,
            "temperature": 10,
            "batch_size": 10,
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
            "save_results_dir": "/media/arpit/datadisk/private/10701/project/results/exp8/"
        }


# run agent
returns, Qs = run_experiment(train_param)

# plot results
plt.figure(1)
plt.plot(range(len(returns)), returns)
plt.xlabel("episode")
plt.ylabel("average return")
plt.title("Average Return")

plt.figure(2)
plt.plot(range(len(Qs)), Qs)
plt.xlabel("episode")
plt.ylabel("mean Q value")
plt.title("Mean Q Value")

plt.show()