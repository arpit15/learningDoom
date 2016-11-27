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

from RestrictedEnvironment import Environment
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

class Agent(object):
    def __init__(self, discount, level, algorithm, prioritized_experience, max_memory, exploration_policy,
                 learning_rate, history_length, batch_size, combine_actions, target_update_freq, epsilon_start, epsilon_end,
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
        self.vgg_feat_num = 512
        self.vgg_feat_shape = 7
        self.use_vgg = True

        # initialization
        self.environment = Environment(level=level, combine_actions=combine_actions, visible=visible)
        self.memory = ExperienceReplay(max_memory=max_memory, prioritized=prioritized_experience, store_episodes=(max_action_sequence_length>1))
        self.preprocessed_curr = []
        self.win_count = 0
        self.curr_step = 0

        self.state_width = image_width
        self.state_height = image_height
        self.scale = self.state_width / float(self.environment.screen_width)

        # recurrent
        self.max_action_sequence_length = max_action_sequence_length
        self.num_actions = len(self.environment.actions)
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
        """
        self.target_network, self.state_encoder, self.target_state_decoder = self.autoencoder()
        self.online_network, _, self.online_state_decoder = self.autoencoder()
        self.state_encoder.load_weights('state_encoder_model_8000.h5')
        self.state_encoder.compile(adam(lr=5e-4), "mse")
        self.predictor = self.predictor_model()
        self.predictor.load_weights('predictor_model_5000.h5')
        self.predictor.compile(adam(lr=5e-4), "mse")
        """
        #TODO: remove commment

    def predictor_model(self):
        input = Input(shape=(200,))

        x = Dense(200, activation='relu')(input)

        encoded_state = Dense(200, activation='relu')(x)

        # action encoder
        action = Input(shape=(3,))
        x = Dense(input_dim=3, output_dim=8, activation='relu')(action)
        encoded_action = Dense(8, activation='relu')(x)

        x = merge([encoded_state, encoded_action], mode='concat')

        x = Dense(200)(x)

        x = ELU()(x)

        x = Dense(200)(x)

        next_state = ELU()(x)

        predictor = Model(input=[input, action], output=next_state)

        predictor.compile(optimizer=adam(lr=5e-4), loss='mse')

        return predictor

    def autoencoder(self):
        a = 1.0
        input_img = Input(shape=(self.history_length, 72, 80))

        # state encoder
        x = Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same', trainable=False)(input_img)
        x = ELU(a)(x)
        # x = BatchNormalization(mode=2)(x)
        x = Convolution2D(32, 3, 3, subsample=(2, 2), border_mode='same', trainable=False)(x)
        x = ELU(a)(x)
        # x = BatchNormalization(mode=2)(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same', trainable=False)(x)
        x = ELU(a)(x)
        # x = BatchNormalization(mode=2)(x)
        x = Flatten()(x)
        encoded_state = Dense(200, trainable=False)(x)
        encoded_state = ELU(a)(encoded_state)
        # encoded_state = Lambda(lambda a: K.greater(a, K.zeros_like(a)), output_shape=(32,))(encoded_state)
        state_encoder = Model(input=input_img, output=encoded_state)

        input_encoded_state = Input(shape=(200,))

        state_value = Dense(256, activation='relu', init='uniform')
        _state_value = state_value(encoded_state)
        __state_value = state_value(input_encoded_state)
        state_value = Dense(1, init='uniform')
        _state_value = state_value(_state_value)
        __state_value = state_value(__state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1),
                             output_shape=(len(self.environment.actions),))
        _state_value = state_value(_state_value)
        __state_value = state_value(__state_value)

        # action advantage tower - A
        action_advantage = Dense(256, activation='relu', init='uniform')
        _action_advantage = action_advantage(encoded_state)
        __action_advantage = action_advantage(input_encoded_state)
        action_advantage = Dense(len(self.environment.actions), init='uniform')
        _action_advantage = action_advantage(_action_advantage)
        __action_advantage = action_advantage(__action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                  output_shape=(len(self.environment.actions),))
        _action_advantage = action_advantage(_action_advantage)
        __action_advantage = action_advantage(__action_advantage)

        # merge to state-action value function Q
        state_action_value = merge([_state_value, _action_advantage], mode='sum')
        __state_action_value = merge([__state_value, __action_advantage], mode='sum')
        model = Model(input=input_img, output=state_action_value)
        model.compile(rmsprop(lr=self.learning_rate), "mse")
        model_decoder = Model(input=input_encoded_state, output=__state_action_value)
        model_decoder.compile(rmsprop(lr=self.learning_rate), "mse")

        return model, state_encoder, model_decoder

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
                output = Dense(len(self.environment.actions))(flatten)
                model = Model(input=input_img, output=output)
                model.compile(rmsprop(lr=self.learning_rate), "mse")
                #model.summary()
            elif network_type == "sequential":
                print("Built a sequential DQN")
                model = Sequential()
                # print self.history_length, self.state_height, self.state_width
                if self.use_vgg
                    model.add(Convolution2D(16, 3, 3, subsample=(2,2), activation='relu', name='conv1_agent', input_shape=(self.history_length*self.vgg_feat_num, self.vgg_feat_shape, self.vgg_feat_shape), init='uniform', trainable=True))
                else:
                    model.add(Convolution2D(16, 8, 8, subsample=(4,4), activation='relu', name='conv1_agent', input_shape=(self.history_length, self.state_height, self.state_width), init='uniform', trainable=True))
                
                model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu', name='conv2_agent', init='uniform', trainable=True))
               
                ## original DQN
                # model.add(Convolution2D(16, 8, 8, subsample=(4,4), activation='relu', name='conv1_agent', input_shape=(self.history_length, self.state_height, self.state_width), init='uniform', trainable=True))
                # model.add(Convolution2D(32, 4, 4, subsample=(2,2), activation='relu', init='conv2_agent', trainable=True))
                
                model.add(Flatten())
                model.add(Dense(512, activation='relu', name='FC1_agent', init='uniform'))
                model.add(Dense(len(self.environment.actions),init='uniform'))

                model.compile(rmsprop(lr=self.learning_rate), "mse")
            elif network_type == "recurrent":
                print("Built a recurrent DQN")
                model = Sequential()
                model.add(TimeDistributed(Convolution2D(16, 3, 3, subsample=(2,2), activation='relu', init='uniform', trainable=True),input_shape=(self.history_length, 1, self.state_height, self.state_width)))
                model.add(TimeDistributed(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu', init='uniform', trainable=True)))
                model.add(TimeDistributed(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu', init='uniform', trainable=True)))
                model.add(TimeDistributed(Convolution2D(128, 3, 3, subsample=(1,1), activation='relu', init='uniform')))
                model.add(TimeDistributed(Convolution2D(256, 3, 3, subsample=(1,1), activation='relu', init='uniform')))
                model.add(TimeDistributed(Flatten()))
                model.add(LSTM(512, activation='relu', init='uniform', unroll=True))
                model.add(Dense(len(self.environment.actions),init='uniform'))
                model.compile(rmsprop(lr=self.learning_rate), "mse")
                #model.summary()
        elif architecture == Architecture.DUELING:
            if network_type == "sequential":
                print("Built a dueling sequential DQN")
                
                if self.use_vgg:
                    input = Input(shape=(self.history_length*self.vgg_feat_num, self.vgg_feat_shape, self.vgg_feat_shape))
                    x = Convolution2D(16, 3, 3, subsample=(2, 2), activation='relu',
                        input_shape=(self.history_length*self.vgg_feat_num, self.vgg_feat_shape, self.vgg_feat_shape), init='uniform',
                        trainable=True)(input)
                else:
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
                state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1), output_shape=(len(self.environment.actions),))(state_value)
                # action advantage tower - A
                action_advantage = Dense(256, activation='relu', init='uniform')(x)
                action_advantage = Dense(len(self.environment.actions), init='uniform')(action_advantage)
                action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(len(self.environment.actions),))(action_advantage)
                # merge to state-action value function Q
                state_action_value = merge([state_value, action_advantage], mode='sum')
                model = Model(input=input, output=state_action_value)
                model.compile(rmsprop(lr=self.learning_rate), "mse")
                #model.summary()
            else:
                print("ERROR: not implemented")
                exit()
        elif architecture == Architecture.SEQUENCE:
            print("Built a recurrent DQN")
            
            state_model_input = Input(shape=(self.history_length, self.state_height, self.state_width))
            state_model = Convolution2D(16, 3, 3, subsample=(2, 2), activation='relu',
                                          input_shape=(self.history_length, self.state_height, self.state_width),
                                          init='uniform', trainable=True)(state_model_input)
            state_model = Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu', init='uniform', trainable=True)(state_model)
            state_model = Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu', init='uniform', trainable=True)(state_model)
            state_model = Convolution2D(128, 3, 3, subsample=(1, 1), activation='relu', init='uniform')(state_model)
            state_model = Convolution2D(256, 3, 3, subsample=(1, 1), activation='relu', init='uniform')(state_model)
            state_model = Flatten()(state_model)
            state_model = Dense(512, activation='relu', init='uniform')(state_model)
            state_model = RepeatVector(self.max_action_sequence_length)(state_model)

            action_model_input = Input(shape=(self.max_action_sequence_length,))
            action_model = Masking(mask_value=self.end_token, input_shape=(self.max_action_sequence_length,))(action_model_input)
            action_model = Embedding(input_dim=self.input_action_space_size, output_dim=100, init='uniform',
                                       input_length=self.max_action_sequence_length)(action_model)
            action_model = TimeDistributed(Dense(100, init='uniform', activation='relu'))(action_model)

            x = merge([state_model, action_model], mode='concat', concat_axis=-1)
            x = LSTM(512, return_sequences=True, activation='relu', init='uniform')(x)

            # state value tower - V
            state_value = TimeDistributed(Dense(256, activation='relu', init='uniform'))(x)
            state_value = TimeDistributed(Dense(1, init='uniform'))(state_value)
            state_value = Lambda(lambda s: K.repeat_elements(s,rep=len(self.environment.actions),axis=2))(state_value)

            # action advantage tower - A
            action_advantage = TimeDistributed(Dense(256, activation='relu', init='uniform'))(x)
            action_advantage = TimeDistributed(Dense(len(self.environment.actions), init='uniform'))(action_advantage)
            action_advantage = TimeDistributed(Lambda(lambda a: a - K.mean(a, keepdims=True, axis=-1)))(action_advantage)

            # merge to state-action value function Q
            state_action_value = merge([state_value, action_advantage], mode='sum')

            model = Model(input=[state_model_input, action_model_input], output=state_action_value)
            model.compile(rmsprop(lr=self.learning_rate), "mse")
            model.summary()

        return model

    
    def get_vgg_feat(self, state):
        x = scipy.misc.imresize(state, size=(224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = imagenet_preprocess(x)
        pred = self.preprocess_model.predict(x)
        # return np.squeeze(pred[0,0,:,:])
        return pred


    def preprocess(self, state):
        # resize image and convert to greyscale
        if self.scale == 1:
            return np.mean(state,0)
        else:
            if self.use_vgg:
                # with vgg features
                state = self.get_vgg_feat(state)
            else:
                # with scaling
                state = scipy.misc.imresize(state, self.scale)

            return state

    def get_inputs_and_targets_for_sequence(self, minibatch):
        """Given a minibatch, extract the inputs and targets for the training according to DQN or DDQN

                :param minibatch: the minibatch to train on
                :return: the inputs, targets and sample weights (for prioritized experience replay)
                """
        # if self.architecture == Architecture.SEQUENCE:
        #    return self.get_inputs_and_targets_for_sequence(minibatch)

        targets = list()
        action_idxs = list()
        inputs = list()
        samples_weights = list()
        for idx, transition_list, game_over, sample_weight in minibatch:

            # choose random end transition from the episode
            end_idx = np.random.randint(0, len(transition_list))
            start_idx = max(0, end_idx - self.max_action_sequence_length + 1)

            # there should be at least one chosen transition)
            chosen_transitions = transition_list[start_idx:end_idx+1]
            num_chosen_transitions = len(chosen_transitions)
            first_transition = chosen_transitions[0]
            last_transition = chosen_transitions[-1]

            # relevant actions
            chosen_actions = [transition.action for transition in chosen_transitions]
            input_actions = [self.start_token] + chosen_actions[:-1]
            # pad in the end if necessary
            if len(input_actions) < self.max_action_sequence_length:
                input_actions += [self.end_token] * (self.max_action_sequence_length - num_chosen_transitions)
            actions_for_next_state = [self.start_token] + [self.end_token] * (self.max_action_sequence_length - 1)

            # prepare input for predicting the current and next actions
            curr_input = [first_transition.preprocessed_curr, np.array([input_actions])]
            next_input = [last_transition.preprocessed_next, np.array([actions_for_next_state])]

            action_idxs.append(input_actions)
            inputs.append(curr_input[0][0])

            # get the current action-values
            target = self.online_network.predict(curr_input)[0]

            # calculate TD-target for last transition
            next_value = 0
            if game_over and end_idx == len(transition_list)-1:
                next_value = last_transition.reward
            else:
                if self.algorithm == Algorithm.DQN:
                    Q_sa = self.target_network.predict(next_input)[0][0]
                    next_value = np.max(Q_sa)

                elif self.algorithm == Algorithm.DDQN:
                    best_next_action = np.argmax(self.online_network.predict(next_input)[0][0])
                    next_value = self.target_network.predict(next_input)[0][0][best_next_action]

            current_index = min(self.max_action_sequence_length, num_chosen_transitions) - 1
            for idx in range(current_index,-1,-1):
                transition = chosen_transitions[idx]
                TD_target = transition.reward + self.discount * next_value
                TD_error = TD_target - target[idx][transition.action]
                target[idx][transition.action] = TD_target

            targets.append(target)

            # updates priority and weight for prioritized experience replay
            if self.memory.prioritized:
                self.memory.update_transition_priority(idx, np.abs(TD_error))
                samples_weights.append(sample_weight)

        #print(action_idxs)
        return np.array(inputs), np.array(targets), np.array(samples_weights), np.array(action_idxs)

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

        if not self.use_vgg:
            return np.array(inputs), np.array(targets), np.array(samples_weights), np.array(action_idxs)
            
        else:
            input_array = np.array(inputs)
            target_array = np.array(targets)

            input_shape = input_array.shape
            
            input_array = np.reshape(input_array,(input_shape[0]*input_shape[1], input_shape[2],input_shape[3]))
            target_array = np.reshape(target_array,(input_shape[0]*input_shape[1], input_shape[2],input_shape[3]))
            
            return input_array, target_array, np.array(samples_weights), np.array(action_idxs)


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
            action_idx = np.random.randint(len(self.environment.actions))
        action = self.environment.actions[action_idx]

        # anneal epsilon value
        if self.epsilon > self.epsilon_end:
            self.epsilon -= float(self.epsilon_start - self.epsilon_end)/float(self.epsilon_annealing_steps)

        return action, action_idx

    def get_action_according_to_exploration_policy(self, Q):
        # action, action_idx = self.environment.actions[0], 0
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

    def predict_sequence(self):
        """predict action according to the current state

        :return: the action, the action index, the mean Q value
        """
        # if no current state is present, create one by stacking the duplicated current state

        if self.preprocessed_curr == []:
            frame = self.environment.get_curr_state()
            preprocessed_frame = self.preprocess(frame)
            for t in range(self.history_length):
                self.preprocessed_curr.append(preprocessed_frame)

        # choose action
        preprocessed_curr = np.reshape(self.preprocessed_curr, (1, self.history_length, self.state_height, self.state_width))

        actions = []
        action_idxs = []
        # predict a single action
        curr_idx = 1
        input_actions = [self.start_token] + [self.end_token] * (self.max_action_sequence_length-1)
        for idx in range(1,self.max_action_sequence_length+1):
            Q = self.online_network.predict([preprocessed_curr, np.array([input_actions])], batch_size=1)[0]
            action_value = Q[idx-1]
            if idx > 1 and np.max(action_value) < last_max_Q:
                break
            last_max_Q = np.max(action_value)
            action, action_idx = self.get_action_according_to_exploration_policy(action_value)
            if idx < self.max_action_sequence_length:
                input_actions[idx] = action_idx
            actions += [action]
            action_idxs += [action_idx]

        return actions, action_idxs, np.max(Q) # send as a list of actions to conform with episodic experience replay

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
        if self.use_vgg:
            preprocessed_curr = np.reshape(self.preprocessed_curr, (1, self.history_length*self.vgg_feat_num, self.vgg_feat_shape, self.vgg_feat_shape))
        else:
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
            frame, r, game_over = self.environment.step(action)
            if self.visible:
                self.environment.show()
            reward += r # reward is accumulated
            if game_over:
                break
            if t == self.skipped_frames-1: # rest are skipped
                preprocessed_next.append(self.preprocess(frame))

        # episode finished
        if game_over:
            preprocessed_next = []
            self.environment.new_episode()
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
    from RestrictedEnvironment import Level
    import numpy as np
    params = {
            "snapshot_episodes": 100,
            "episodes": 5,
            "steps_per_episode": 300, # 4300 for deathmatch, 300 for health gathering
            "average_over_num_episodes": 50,
            "start_learning_after": 20,
            "algorithm": Algorithm.DDQN,
            "discount": 0.99,
            "max_memory": 50000,
            "prioritized_experience": True,
            "exploration_policy": ExplorationPolicy.E_GREEDY,
            "learning_rate": 2.5e-4,
            "level": Level.HEALTH,
            "combine_actions": False,
            "temperature": 10,
            "batch_size": 32,
            "history_length": 4,
            "snapshot": 'exp8_5000.h5',#result_dir + 'model_20.h5',
            "snapshot_itr_num": 0,
            "mode": Mode.DISPLAY,
            "skipped_frames": 4,
            "target_update_freq": 1000,
            "steps_between_train": 1,
            "epsilon_start": 0.5,
            "epsilon_end": 0.01,
            "epsilon_annealing_steps": 3e4,
            "architecture": Architecture.DIRECT,
            "max_action_sequence_length": 1,
            "save_results_dir": '',
            "visible": True
        }
    agent = Agent(algorithm=params["algorithm"],
                  discount=params["discount"],
                  snapshot=params["snapshot"],
                  max_memory=params["max_memory"],
                  prioritized_experience=params["prioritized_experience"],
                  exploration_policy=params["exploration_policy"],
                  learning_rate=params["learning_rate"],
                  level=params["level"],
                  history_length=params["history_length"],
                  batch_size=params["batch_size"],
                  temperature=params["temperature"],
                  combine_actions=params["combine_actions"],
                  train=(params["mode"] == Mode.TRAIN),
                  skipped_frames=params["skipped_frames"],
                  target_update_freq=params["target_update_freq"],
                  epsilon_start=params["epsilon_start"],
                  epsilon_end=params["epsilon_end"],
                  epsilon_annealing_steps=params["epsilon_annealing_steps"],
                  architecture=params["architecture"],
                  visible=params["visible"],
                  max_action_sequence_length=params["max_action_sequence_length"])

    model = agent.target_network
    
    #visualizing the conv filters
    conv_layer = 5
    for l in range(conv_layer):
        l1 = model.layers[l].get_weights()
        w1 = np.asarray(l1[0])
        
        f, axarr = plt.subplots(4,4)
        for i in range(4):
            for j in range(4):
                axarr[i,j].imshow(np.squeeze(w1[i,j,:,:]), cmap='Greys_r')

    #     plt.savefig("layer_" + str(l) + ".png", bbox_inches="tight")
    set_trace()

    #run for some steps
    for i in range(80):
        actions, action_idxs, mean_Q = agent.predict()
        for action, action_idx in zip(actions, action_idxs):
            action_idx = int(action_idx)
            next_state, reward, game_over = agent.step(action, action_idx)
            agent.environment.show()

    #visualizing the output of the conv filters
    agent.predict()
    preprocessed_curr = np.reshape(agent.preprocessed_curr, (1, agent.history_length, agent.state_height, agent.state_width))
    input_image = np.copy(preprocessed_curr)
    input_image = input_image.astype(np.float32)
    f, axarr = plt.subplots(1,4)
    
    # plotting input image
    for j in range(4):
       axarr[j].imshow(np.squeeze(input_image[0,j,:,:]), cmap='Greys_r')
       axarr[j].set_axis_off()
    plt.show()

    # plotting conv layer output
    for l in range(conv_layer):
        get_lth_layer_output = K.function([model.layers[l].input],
                                  [model.layers[l].output])
    
        layer_output = get_lth_layer_output([input_image,0])[0]
        out = np.squeeze(layer_output)
        # set_trace()
        m = int(min(4,layer_output.shape[1]/4.0))
        f, axarr = plt.subplots(4,m)
        
        for i in range(4):
            for j in range(4):
                axarr[i,j].set_axis_off()
                axarr[i,j].imshow(np.squeeze(out[i+j*4,:,:]), cmap='Greys_r')

        plt.savefig("layer_output_" + str(l) + ".png", bbox_inches="tight")

        input_image = np.copy(layer_output)



    