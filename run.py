import numpy as np
import datetime
from agent import Agent
from config import Mode
from timeit import timeit
# import matplotlib.pyplot as plt

# from pdb import set_trace

def run_experiment(args):
    """ Run a single experiment, either train, test or display of an agent

    :param args: a dictionary containing all the parameters for the run
    :return: lists of average returns and mean Q values
    """
    global resultDir
    resultDir = args["save_results_dir"]
    agent = Agent(algorithm=args["algorithm"],
                  discount=args["discount"],
                  snapshot=args["snapshot"],
                  max_memory=args["max_memory"],
                  prioritized_experience=args["prioritized_experience"],
                  exploration_policy=args["exploration_policy"],
                  learning_rate=args["learning_rate"],
                  level=args["level"],
                  history_length=args["history_length"],
                  batch_size=args["batch_size"],
                  temperature=args["temperature"],
                  combine_actions=args["combine_actions"],
                  train=(args["mode"] == Mode.TRAIN),
                  skipped_frames=args["skipped_frames"],
                  target_update_freq=args["target_update_freq"],
                  epsilon_start=args["epsilon_start"],
                  epsilon_end=args["epsilon_end"],
                  epsilon_annealing_steps=args["epsilon_annealing_steps"],
                  architecture=args["architecture"],
                  visible=args["visible"],
                  max_action_sequence_length=args["max_action_sequence_length"])

    if (args["mode"] == Mode.TEST or args["mode"] == Mode.DISPLAY) and args["snapshot"] == '':
        print("Warning: mode set to " + str(args["mode"]) + " but no snapshot was loaded")

    n = float(args["average_over_num_episodes"])

    # initialize
    total_steps = 0
    returns_over_all_episodes = []
    mean_q_over_all_episodes = []
    return_buffer = []
    mean_q_buffer = []
    
    start_time = timeit()
    
    #plotting init
    # plt.ion()
    # f, axarr = plt.subplots(2, sharex=True)
    # plt.xlim([1,args["episodes"]])
    # axarr[0].set_title("Training performance")
    # axarr[0].set_ylabel('Average Return')
    # axarr[0].set_xlabel("episode")
    # axarr[1].set_ylabel('Average Q value')
    # axarr[1].set_xlabel("episode")
    # plt.draw()
    # plt.savefig('alldone.png', dpi=300)
    # set_trace()
    # axarr[0].plot(range(10000), range(10000))

    for i in range(args["episodes"]):
        
        agent.environment.new_episode()
        steps, curr_return, curr_Qs, loss = 0, 0, 0, 0
        game_over = False
        while not game_over and steps < args["steps_per_episode"]:
            #print("predicting")
            actions, action_idxs, mean_Q = agent.predict()
            for action, action_idx in zip(actions, action_idxs):
                action_idx = int(action_idx)
                next_state, reward, game_over = agent.step(action, action_idx)
                agent.store_next_state(next_state, reward, game_over, action_idx)
                steps += 1
                total_steps += 1
                curr_return += reward
                curr_Qs += mean_Q

                if i > args["start_learning_after"] and args["mode"] == Mode.TRAIN and total_steps % args["steps_between_train"] == 0:
                    loss += agent.train()
                    # print(">>>finished training after:"+str(total_steps))

                if game_over or steps > args["steps_per_episode"]:
                    break

        # store stats
        if len(return_buffer) > n:
            del return_buffer[0]
        return_buffer += [curr_return]
        average_return = np.mean(return_buffer)

        if len(mean_q_buffer) > n:
            del mean_q_buffer[0]
        mean_q_buffer += [curr_Qs / float(steps)]
        average_mean_q = np.mean(mean_q_buffer) #mean_{actions} Q_{episodeMean}

        returns_over_all_episodes += [average_return]
        mean_q_over_all_episodes += [average_mean_q]

        print("")
        print(str(datetime.datetime.now()))
        print("episode = " + str(i) + " steps = " + str(total_steps))
        print("epsilon = " + str(agent.epsilon) + " loss = " + str(loss))
        print("current_return = " + str(curr_return) + " average return = " + str(average_return))
        
        # save snapshot of target network
        if args["mode"] == Mode.TRAIN and i % args["snapshot_episodes"] == args["snapshot_episodes"] - 1:
            snapshot = 'model_' + str(i + 1 + args["snapshot_itr_num"]) + '.h5'
            snapshot = resultDir + snapshot
            print(str(datetime.datetime.now()) + " >> saving snapshot to " + snapshot)
            agent.target_network.save_weights(snapshot, overwrite=True)


        #plotting
        # print("plotting")
        # axarr[0].plot(range(len(returns_over_all_episodes)), returns_over_all_episodes)
        # axarr[1].plot(range(len(mean_q_over_all_episodes)), mean_q_over_all_episodes)
        # plt.show()
        # plt.savefig('alldone.png', dpi=300)


    print("time for this episode:"+str((timeit()-start_time)/args["episodes"]))
    agent.environment.game.close()

    if args["save_ERM"] != '':
        agent.memory.save(args["save_ERM"])    
        print 'saving ERM'
    return returns_over_all_episodes, mean_q_over_all_episodes


# if __name__ == "__main__":
#     experiment = "single_agent" # TODO: create a better way for this

#     if experiment == "multi_agent":
#         # multi agent entity

#         aiming_agent = {
#             "algorithm": Algorithm.DDQN,
#             "discount": 0.99,
#             "max_memory": 10000,
#             "prioritized_experience": False,
#             "exploration_policy": ExplorationPolicy.E_GREEDY,
#             "learning_rate": 2.5e-4,
#             "level": Level.DEFEND,
#             "combine_actions": True,
#             "temperature": 10,
#             "batch_size": 10,
#             "history_length": 4,
#             "snapshot": 'defend_model_1000.h5',
#             "mode": Mode.TRAIN,
#             "skipped_frames": 4,
#             "target_update_freq": 3000
#         }

#         exploring_agent = {
#             "algorithm": Algorithm.DDQN,
#             "discount": 0.99,
#             "max_memory": 10000,
#             "prioritized_experience": False,
#             "exploration_policy": ExplorationPolicy.E_GREEDY,
#             "learning_rate": 2.5e-4,
#             "level": Level.HEALTH,
#             "combine_actions": True,
#             "temperature": 10,
#             "batch_size": 10,
#             "history_length": 4,
#             "snapshot": 'health_model_500.h5',
#             "mode": Mode.TRAIN,
#             "skipped_frames": 4,
#             "target_update_freq": 3000
#         }

#         entity_args = {
#             "snapshot_episodes": 1000,
#             "episodes": 2000,
#             "steps_per_episode": 4000,  # 4300 for deathmatch, 300 for health gathering
#             "average_over_num_episodes": 50,
#             "start_learning_after": 200,
#             "mode": Mode.TRAIN,
#             "history_length": 4,
#             "level": Level.DEATHMATCH,
#             "combine_actions": True
#         }

#         entity = Entity([aiming_agent, exploring_agent], entity_args)
#         returns = entity.run()

#         plt.plot(range(len(returns)), returns, "r")
#         plt.xlabel("episode")
#         plt.ylabel("average return")
#         plt.title("Average Return")

#     elif experiment == "single_agent":
        # lstm = {
        #     "snapshot_episodes": 100,
        #     "episodes": 1500,
        #     "steps_per_episode": 400,  # 4300 for deathmatch, 300 for health gathering
        #     "average_over_num_episodes": 50,
        #     "start_learning_after": 30,
        #     "algorithm": Algorithm.DDQN,
        #     "discount": 0.99,
        #     "max_memory": 5000,
        #     "prioritized_experience": False,
        #     "exploration_policy": ExplorationPolicy.SOFTMAX,
        #     "learning_rate": 2.5e-4,
        #     "level": Level.DEFEND,
        #     "combine_actions": True,
        #     "temperature": 10,
        #     "batch_size": 10,
        #     "history_length": 4,
        #     "snapshot": '',
        #     "mode": Mode.TRAIN,
        #     "skipped_frames": 7,
        #     "target_update_freq": 1000,
        #     "steps_between_train": 1,
        #     "epsilon_start": 0.7,
        #     "epsilon_end": 0.01,
        #     "epsilon_annealing_steps": 3e4,
        #     "architecture": Architecture.DIRECT
        # }

        # egreedy = {
        #     "snapshot_episodes": 100,
        #     "episodes": 400,
        #     "steps_per_episode": 40, # 4300 for deathmatch, 300 for health gathering
        #     "average_over_num_episodes": 50,
        #     "start_learning_after": 20,
        #     "algorithm": Algorithm.DDQN,
        #     "discount": 0.99,
        #     "max_memory": 1000,
        #     "prioritized_experience": False,
        #     "exploration_policy": ExplorationPolicy.E_GREEDY,
        #     "learning_rate": 2.5e-4,
        #     "level": Level.BASIC,
        #     "combine_actions": True,
        #     "temperature": 10,
        #     "batch_size": 10,
        #     "history_length": 4,
        #     "snapshot": '',
        #     "mode": Mode.TRAIN,
        #     "skipped_frames": 4,
        #     "target_update_freq": 1000,
        #     "steps_between_train": 1,
        #     "epsilon_start": 0.5,
        #     "epsilon_end": 0.01,
        #     "epsilon_annealing_steps": 3e4,
        #     "architecture": Architecture.DIRECT,
        #     "max_action_sequence_length": 1
        # }

        # lstm = {
        #     "snapshot_episodes": 100,
        #     "episodes": 6000,
        #     "steps_per_episode": 400, # 4300 for deathmatch, 300 for health gathering
        #     "average_over_num_episodes": 50,
        #     "start_learning_after": 10,
        #     "algorithm": Algorithm.DDQN,
        #     "discount": 0.99,
        #     "max_memory": 1000,
        #     "prioritized_experience": False,
        #     "exploration_policy": ExplorationPolicy.E_GREEDY,
        #     "learning_rate": 2.5e-4,
        #     "level": Level.DEATHMATCH,
        #     "combine_actions": True,
        #     "temperature": 10,
        #     "batch_size": 10,
        #     "history_length": 4,
        #     "snapshot": '',
        #     "mode": Mode.TRAIN,
        #     "skipped_frames": 4,
        #     "target_update_freq": 1000,
        #     "steps_between_train": 1,
        #     "epsilon_start": 0.5,
        #     "epsilon_end": 0.01,
        #     "epsilon_annealing_steps": 3e4,
        #     "architecture": Architecture.SEQUENCE,
        #     "max_action_sequence_length": 5
        # }

        # runs = [lstm]

        # colors = ["r", "g", "b"]
        # for color, run in zip(colors, runs):
        #     # run agent
        #     returns, Qs = run_experiment(run)

        #     # plot results
        #     plt.figure(1)
        #     plt.plot(range(len(returns)), returns, color)
        #     plt.xlabel("episode")
        #     plt.ylabel("average return")
        #     plt.title("Average Return")

        #     plt.figure(2)
        #     plt.plot(range(len(Qs)), Qs, color)
        #     plt.xlabel("episode")
        #     plt.ylabel("mean Q value")
        #     plt.title("Mean Q Value")

        # plt.show()
