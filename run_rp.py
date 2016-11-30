import numpy as np
import datetime
from agent_rp import Agent
from config import Mode
from time import time
import tensorflow as tf

import os
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.25):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



# import matplotlib.pyplot as plt

# from pdb import set_trace

def run_experiment(args):
    """ Run a single experiment, either train, test or display of an agent

    :param args: a dictionary containing all the parameters for the run
    :return: lists of average returns and mean Q values
    """
    KTF.set_session(get_session())
    global resultDir
    do_logging = False
    if "log_dir" in args:
        do_logging = True
        train_writer = tf.train.SummaryWriter(args["log_dir"]+"summary/")

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
    
    if args["load_ERM"]:
        agent.memory = args["ERM"]

    if (args["mode"] == Mode.TEST or args["mode"] == Mode.DISPLAY) and args["snapshot"] == '':
        print("Warning: mode set to " + str(args["mode"]) + " but no snapshot was loaded")

    n = float(args["average_over_num_episodes"])

    # initialize
    total_steps = 0
    returns_over_all_episodes = []
    mean_q_over_all_episodes = []
    return_buffer = []
    mean_q_buffer = []
    
    

    for i in range(args["episodes"]):
        start_time = time()
        agent.environment.new_episode()
        steps, curr_return, curr_Qs, loss = 0, 0, 0, 0
        game_over = False
        print len(agent.memory_non_zero.memory),
        print len(agent.memory_zero.memory)
        while not game_over and steps < args["steps_per_episode"]:
            #print("predicting")
            actions, action_idxs, mean_Q = agent.predict()
            for action, action_idx in zip(actions, action_idxs):
                action_idx = int(action_idx)
                # print action
                next_state, reward, game_over = agent.step(action, action_idx)
                agent.store_next_state(next_state, reward, game_over, action_idx)
                steps += 1
                total_steps += 1
                curr_return += reward
                curr_Qs += mean_Q

                if len(agent.memory_non_zero.memory)!=0 and len(agent.memory_zero.memory)!=0 and i > args["start_learning_after"] and args["mode"] == Mode.TRAIN and total_steps % args["steps_between_train"] == 0:
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

        
        # add to tensorboard summary
        if do_logging:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = average_mean_q
            summary_value.tag = 'q'
            train_writer.add_summary(summary, i+1)

            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = average_return
            summary_value.tag = 'reward'
            train_writer.add_summary(summary, i+1)

        # flush to memory
        if do_logging and ((i+1)%args["log_after_episodes"] == 0):
            train_writer.flush()


        print("")
        print(str(datetime.datetime.now()))
        print("episode = " + str(i) + " steps = " + str(steps))
        print("epsilon = " + str(agent.epsilon) + " loss = " + str(loss))
        print("current_return = " + str(curr_return) + " average return = " + str(average_return))
        
        # save snapshot of target network
        if args["mode"] == Mode.TRAIN and i % args["snapshot_episodes"] == args["snapshot_episodes"] - 1:
            snapshot = 'models/model_' + str(i + 1 + args["snapshot_itr_num"]) + '.h5'
            snapshot = resultDir + snapshot
            print(str(datetime.datetime.now()) + " >> saving snapshot to " + snapshot)
            agent.target_network.save_weights(snapshot, overwrite=True)


        print("time for this episode:"+str((time()-start_time)))
    agent.environment.game.close()

    if args["save_ERM"] != '':
        agent.memory.save(args["save_ERM"])    
        print 'saving ERM'
    return returns_over_all_episodes, mean_q_over_all_episodes ,agent.memory_non_zero
