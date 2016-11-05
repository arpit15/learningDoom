#author: Arpit Agarwal

import gym
from enum import Enum
import sys
sys.dont_write_bytecode = True

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
        self.actions_num = self.game.action_space.shape

        if self.combine_actions:
            for perm in it.product([False, True], repeat=self.actions_num):
                self.actions.append(list(perm))
        else:
            for action in range(self.actions_num):
                one_hot = [False] * self.actions_num
                one_hot[action] = True
                self.actions.append(one_hot)

        self.screen_width = self.game.screen_width
        self.screen_height = self.game.screen_height

    def step(self, action):
        next_state, reward, game_over, _ = self.game.step(action)
        return next_state, reward, game_over

    def get_curr_state(self):
        action = [0]*self.actions_num
        curr_state, _, _, _ = self.game.step(action)
        return curr_state

    def new_episode(self):
        self.game.reset()

    def is_game_over(self):
        return self.game.is_episode_finished()


