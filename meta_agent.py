import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam, rmsprop
from keras.layers.core import RepeatVector, Masking, TimeDistributedDense, Reshape
from keras.initializations import uniform
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import scipy.ndimage
from time import sleep
import matplotlib.pyplot as plt
import itertools as it

# from enum import Enum

from agent import Agent
from config import *
from ERM import *

# preprocessing
from vgg_feat import VGG16
from imagenet_utils import preprocess_input as imagenet_preprocess
from keras.preprocessing import image

import sys
sys.dont_write_bytecode = True

from timeit import timeit
from pdb import set_trace

resultDir = "/media/arpit/datadisk/private/10701/project/results/"
image_height, image_width = 60, 80 #TODO: change to 72
merged_model = []

class MetaAgent(object):
    def __init__(self, discount, level, algorithm, prioritized_experience, max_memory, exploration_policy,
                 learning_rate, history_length, batch_size, target_update_freq, epsilon_start, epsilon_end,
                 epsilon_annealing_steps, temperature=10, snapshot='', train=True, visible=True, skipped_frames=4,
                 architecture=Architecture.DIRECT, max_action_sequence_length=1):

        self.trainable = train
        self.visible = visible
        # e-greedy policy
        self.epsilon_annealing_steps = epsilon_annealing_steps #steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if self.trainable:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = self.epsilon_end

        # softmax / multinomial policy
        self.average_minimum = 0 # for multinomial policy
        self.temperature = temperature

        self.policy = exploration_policy

        # preprocessing
        self.preprocess_model = VGG16()


        # initialization
        self.agent = []
        self.memory = ExperienceReplay(max_memory=max_memory, prioritized=prioritized_experience, store_episodes=(max_action_sequence_length>1))
        self.preprocessed_curr = []
        self.win_count = 0
        self.curr_step = 0

        self.state_width = image_width
        self.state_height = image_height
        self.scale = self.state_width / float(self.agent[0].environment.screen_width)

        # recurrent
        self.max_action_sequence_length = max_action_sequence_length
        self.num_actions = len(self.agent[0].environment.actions)
        self.input_action_space_size = self.num_actions + 2 # number of actions + start and end (padding) tokens
        self.output_action_space_size = self.num_actions
        self.start_token = self.num_actions
        self.end_token = self.num_actions + 1

        # training
        self.discount = discount
        self.history_length = history_length # should be 1 for DRQN
        self.skipped_frames = skipped_frames
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.incremental_target_update = False
        self.increment_each_num_steps = 10
        self.tau = 30/float(self.target_update_freq)

        self.algorithm = algorithm
        self.architecture = architecture

        self.target_network = self.create_network(architecture=architecture, algorithm=algorithm)
        self.online_network = self.create_network(architecture=architecture, algorithm=algorithm)
        if snapshot != '':
            print("loading snapshot " + str(snapshot))
            self.target_network.load_weights(snapshot)
            self.online_network.load_weights(snapshot)
            self.target_network.compile(adam(lr=self.learning_rate), "mse")
            self.online_network.compile(adam(lr=self.learning_rate), "mse")
        

    def create_network(self, architecture=Architecture.DIRECT, algorithm=Algorithm.DDQN):
        if algorithm == Algorithm.DRQN:
            network_type = "recurrent"
        else:
            network_type = "sequential"

        if architecture == Architecture.DIRECT:
            if network_type == "inception":
                print("Built an inception DQN")
                input_img = Input(shape=(self.history_length, self.state_height, self.state_width))
                tower_1 = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(input_img)
                tower_1 = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(tower_1)
                tower_2 = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(input_img)
                tower_2 = Convolution2D(16, 5, 5, border_mode='same', activation='relu')(tower_2)
                tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
                tower_3 = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(tower_3)
                output1 = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)
                avgpool = AveragePooling2D((7, 7), strides=(8, 8))(output1)
                flatten = Flatten()(avgpool)
                output = Dense(len(self.num_actions))(flatten)
                model = Model(input=input_img, output=output)
                model.compile(rmsprop(lr=self.learning_rate), "mse")
                #model.summary()
            elif network_type == "sequential":
                print("Built a sequential DQN")
                model = Sequential()

                model.add(Convolution2D(16, 8, 8, subsample=(4,4), activation='relu', name='conv1_meta_agent', input_shape=(self.history_length, self.state_height, self.state_width), init='uniform', trainable=True))
                model.add(Convolution2D(32, 4, 4, subsample=(2,2), activation='relu', init='conv2_meta_agent', trainable=True))
                
                model.add(Flatten())
                model.add(Dense(512, activation='relu', name='FC1_meta_agent', init='uniform'))
                model.add(Dense(len(self.num_actions),init='uniform'))

                model.compile(rmsprop(lr=self.learning_rate), "mse")
            elif network_type == "recurrent":
                pass

        elif architecture == Architecture.DUELING:
            if network_type == "sequential":
                print("Built a dueling sequential DQN")
                input = Input(shape=(self.history_length, self.state_height, self.state_width))
                x = Convolution2D(16, 3, 3, subsample=(2, 2), activation='relu',
                    input_shape=(self.history_length, image_height, image_width), init='uniform',
                    trainable=True)(input)
                x = Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu', init='uniform', trainable=True)(x)
                x = Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu', init='uniform', trainable=True)(x)
                x = Convolution2D(128, 3, 3, subsample=(1, 1), activation='relu', init='uniform')(x)
                x = Convolution2D(256, 3, 3, subsample=(1, 1), activation='relu', init='uniform')(x)
                x = Flatten()(x)
                # state value tower - V
                state_value = Dense(256, activation='relu', init='uniform')(x)
                state_value = Dense(1, init='uniform')(state_value)
                state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1), output_shape=(len(self.num_actions),))(state_value)
                # action advantage tower - A
                action_advantage = Dense(256, activation='relu', init='uniform')(x)
                action_advantage = Dense(len(self.num_actions), init='uniform')(action_advantage)
                action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(len(self.num_actions),))(action_advantage)
                # merge to state-action value function Q
                state_action_value = merge([state_value, action_advantage], mode='sum')
                model = Model(input=input, output=state_action_value)
                model.compile(rmsprop(lr=self.learning_rate), "mse")
                #model.summary()
            else:
                print("ERROR: not implemented")
                exit()
        elif architecture == Architecture.SEQUENCE:
            pass
        return model

    
    def get_vgg_feat(self, state):
        x = scipy.misc.imresize(state, size=(224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = imagenet_preprocess(x)
        pred = self.preprocess_model.predict(x)
        return np.squeeze(pred[0,0,:,:])

    def preprocess(self, state):
        # resize image and convert to greyscale
        if self.scale == 1:
            return np.mean(state,0)
        else:
            state_vgg = self.get_vgg_feat(state)
            state = scipy.misc.imresize(state_vgg, (self.state_height, self.state_width))
            #state = np.lib.pad(state, ((6, 6), (0, 0)), 'constant', constant_values=(0)) #TODO: remove comment
            return state


    def get_inputs_and_targets(self, minibatch):
        """Given a minibatch, extract the inputs and targets for the training according to DQN or DDQN

        :param minibatch: the minibatch to train on
        :return: the inputs, targets and sample weights (for prioritized experience replay)
        """
        if self.architecture == Architecture.SEQUENCE:
            return self.get_inputs_and_targets_for_sequence(minibatch)

        targets = list()
        action_idxs = list()
        inputs = list()
        samples_weights = list()
        for idx, transition_list, game_over, sample_weight in minibatch:
            # for episodic experience - choose a random ending action
            transition = transition_list[0]
            inputs.append(transition.preprocessed_curr[0])

            # prepare input for predicting the current and next actions
            curr_input = transition.preprocessed_curr
            next_input = transition.preprocessed_next

            # get the current action-values
            target = self.online_network.predict(curr_input)[0]

            # calculate TD-target for last transition
            if game_over:
                TD_target = transition.reward
            else:
                if self.algorithm == Algorithm.DQN:
                    Q_sa = self.target_network.predict(next_input)
                    TD_target = transition.reward + self.discount * np.max(Q_sa)

                elif self.algorithm == Algorithm.DDQN:
                    best_next_action = np.argmax(self.online_network.predict(next_input))
                    Q_sa = self.target_network.predict(next_input)[0][best_next_action]
                    TD_target = transition.reward + self.discount * Q_sa

            TD_error = TD_target - target[transition.action]
            target[transition.action] = TD_target
            targets.append(target)

            # updates priority and weight for prioritized experience replay
            if self.memory.prioritized:
                self.memory.update_transition_priority(idx, np.abs(TD_error))
                samples_weights.append(sample_weight)

        return np.array(inputs), np.array(targets), np.array(samples_weights), np.array(action_idxs)

    def softmax_selection(self, Q):
        """Select the action according to the softmax exploration policy

        :param Q: the Q values for the current state
        :return: the action and the action index
        """
        # compute thresholds and choose a random number
        print Q
        exp_Q = np.array(np.exp(Q/float(self.temperature)), copy=True)
        prob = np.random.rand(1)
        importances = [action_value/float(np.sum(exp_Q)) for action_value in exp_Q]
        thresholds = np.cumsum(importances)
        # multinomial sampling according to priorities
        for action_idx, threshold in zip(range(len(thresholds)), thresholds):
            if prob < threshold:
                action = self.environment.actions[action_idx]
                return action, action_idx
        return self.environment.actions[len(exp_Q)-1], len(exp_Q)-1

    def shifted_multinomial_selection(self, Q):
        """Select the action according to a shifted multinomial sampling policy

        :param Q: the Q values of the current state
        :return: the action and the action index
        """
        # Q values are shifted so that we won't have negative values
        self.average_minimum = 0.95 * self.average_minimum + 0.05 * np.min(Q)
        shifted_Q = np.array(Q - min(self.average_minimum, np.min(Q)), copy=True)
        # compute thresholds and choose a random number
        prob = np.random.rand(1)
        importances = [action_value/float(np.sum(shifted_Q)) for action_value in shifted_Q]
        thresholds = np.cumsum(importances)
        # multinomial sampling according to priorities
        for action_idx, threshold in zip(range(len(thresholds)), thresholds):
            if prob < threshold:
                action = self.environment.actions[action_idx]
                return action, action_idx

    def e_greedy(self, Q):
        """ Select the action according to the e-greedy exploration policy

        :param Q: the Q values for the current state
        :return: the action and the action index
        """
        # choose action randomly or greedily
        coin_toss = np.random.rand(1)[0]
        if coin_toss > self.epsilon:
            action_idx = np.argmax(Q)
        else:
            action_idx = np.random.randint(len(self.num_actions))
        action = self.environment.actions[action_idx]

        # anneal epsilon value
        if self.epsilon > self.epsilon_end:
            self.epsilon -= float(self.epsilon_start - self.epsilon_end)/float(self.epsilon_annealing_steps)

        return action, action_idx

    def get_action_according_to_exploration_policy(self, Q):
        
        if self.policy == ExplorationPolicy.E_GREEDY:
            action, action_idx = self.e_greedy(Q)
        elif self.policy == ExplorationPolicy.SHIFTED_MULTINOMIAL:
            action, action_idx = self.shifted_multinomial_selection(Q)
        elif self.policy == ExplorationPolicy.SOFTMAX:
            action, action_idx = self.softmax_selection(Q)
        else:
            print("Error: exploration policy not available")
            exit()
        return action, action_idx

    def predict(self):
        """predict action according to the current state

        :return: the action, the action index, the mean Q value
        """
        # if no current state is present, create one by stacking the duplicated current state
        if self.architecture == Architecture.SEQUENCE:
            return self.predict_sequence()

        if self.preprocessed_curr == []:
            frame = self.environment.get_curr_state()
            preprocessed_frame = self.preprocess(frame)
            for t in range(self.history_length):
                self.preprocessed_curr.append(preprocessed_frame)

        # choose action
        preprocessed_curr = np.reshape(self.preprocessed_curr, (1, self.history_length, self.state_height, self.state_width))
        if self.algorithm == Algorithm.DRQN:
            # expand dims to have a time dimension + switch between depth and time
            preprocessed_curr = np.expand_dims(preprocessed_curr, axis=0).transpose(0,2,1,3,4)

        # predict a single action
        Q = self.online_network.predict(preprocessed_curr, batch_size=1)
        action, action_idx = self.get_action_according_to_exploration_policy(Q)

        return [action], [action_idx], np.max(Q) # send as a list of actions to conform with episodic experience replay

    def step(self, action, action_idx):
        # repeat action several times and stack the first frame onto the previous state
        reward = 0
        game_over = False
        preprocessed_next = list(self.preprocessed_curr)
        del preprocessed_next[0]
        for t in range(self.skipped_frames):
            
            frame, r, game_over = self.agent.step(action,action_idx)
            if self.visible:
                self.agent.environment.show()
            reward += r # reward is accumulated
            if game_over:
                break
            if t == self.skipped_frames-1: # rest are skipped
                preprocessed_next.append(self.preprocess(frame))

        # episode finished
        if game_over:
            preprocessed_next = []
            self.agent[0].environment.new_episode()
            if reward > 0:
                self.win_count += 1 # irrelevant to most levels

        return preprocessed_next, reward, game_over

    def store_next_state(self, preprocessed_next, reward, game_over, action_idx):
        preprocessed_curr = np.reshape(self.preprocessed_curr, (1, self.history_length, image_height, image_width))
        self.preprocessed_curr = list(preprocessed_next) # saved as list
        if preprocessed_next != []:
            preprocessed_next = np.reshape(preprocessed_next, (1, self.history_length, image_height, image_width))

        # store transition
        self.memory.remember(Transition(preprocessed_curr, action_idx, reward, preprocessed_next), game_over) # stored as np array

        self.curr_step += 1

        # update target network with online network once in a while
        if self.incremental_target_update:
            if self.curr_step % self.increment_each_num_steps == 0:
                #print(">>> update the target")
                online_weights = self.online_network.get_weights()
                target_weights = self.target_network.get_weights()
                for i in xrange(len(online_weights)):
                    #print(online_weights[i].shape)
                    #print(target_weights[i].shape)

                    target_weights[i] = self.tau * online_weights[i] + (1 - self.tau) * target_weights[i]
                self.target_network.set_weights(target_weights)
        else:
            if self.curr_step % self.target_update_freq == 0:
                print(">>> update the target")
                self.target_network.set_weights(self.online_network.get_weights())

        return reward, game_over

    def train(self):
        """Train the online network on a minibatch

        :return: the train loss
        """
        
        minibatch = self.memory.sample_minibatch(self.batch_size)
        inputs, targets, samples_weights, action_idxs = self.get_inputs_and_targets(minibatch)
        if self.memory.prioritized:
            return self.online_network.train_on_batch(inputs, targets, sample_weight=samples_weights)
        elif self.architecture == Architecture.SEQUENCE: # episodic
            return self.online_network.train_on_batch([inputs, action_idxs], targets)
        else:
            return self.online_network.train_on_batch(inputs, targets)



if __name__ == "__main__":
    pass