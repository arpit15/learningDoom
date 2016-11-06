#author: Arpit Agarwal

import gym
from enum import Enum
import itertools as it
from time import sleep

import sys
sys.dont_write_bytecode = True

from pdb import set_trace

class Level(Enum):

    BASIC = "DoomBasic-v0"
    HEALTH = "DoomHealthGathering-v0"
    DEATHMATCH = "DoomDeathmatch-v0"
    DEFEND = "DoomDefendCenter-v0"
    WAY_HOME = "DoomMyWayHome-v0"

class Environment(object):
    def __init__(self, level = Level.BASIC, combine_actions = False, visible = True):
        
        self.combine_actions = combine_actions
        self.actions = []

        self.game = gym.make(level.value)
        self.game.reset()

        self.actions_num = len(self.game.allowed_actions)
        self.action_idx = self.game.allowed_actions

        print self.actions_num, self.action_idx

        if self.combine_actions:
            for perm in it.product([False, True], repeat=self.actions_num):
                self.actions.append(list(perm))
        else:
            for action in range(self.actions_num):
                one_hot = [False] * self.actions_num
                one_hot[action] = True
                self.actions.append(one_hot)

        print 'actions combined'
        self.screen_width = self.game.screen_width
        self.screen_height = self.game.screen_height

        if visible:
            self.game.mode = 'normal'
            
    def show(self, close=False):
        self.game.render()

    def remap_action(self, action):
        #transfer from predicted_action space to env.action_space
        remapped_action = [0]*self.game.action_space.shape
        for i in range(self.actions_num):
            remapped_action[self.action_idx[i]] = 1*action[i]

        return remapped_action

    def step(self, action):
        remapped_action = self.remap_action(action)
        # print remapped_action
        next_state, reward, game_over, _ = self.game.step(remapped_action)
        return next_state, reward, game_over

    def get_curr_state(self):
        action = [0]*self.game.action_space.shape
        curr_state, _, _, _ = self.game.step(action)
        return curr_state

    def new_episode(self):
        self.game.reset()

    def is_game_over(self):
        return self.game.is_episode_finished()

    def sample_action(self):
        return self.game.action_space.sample()


if __name__ == "__main__":
    
    env = Environment(level = Level.DEATHMATCH, combine_actions = False, visible = True)
    num_episode = 5
    num_step = 50
    
    for i in range(num_episode):
        print 'episode num:' + str(i+1)
        env.new_episode()
        for j in range(num_step):
            env.show()
            action =  env.sample_action()
            ob, reward, done = env.step(action)
            if done:
                break
            
        sleep(2.0)

    env.show(close=True)