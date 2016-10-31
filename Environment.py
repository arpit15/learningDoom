#author: Arpit Agarwal

import gym
from enum import Enum
import sys
sys.dont_write_bytecode = True

class Level(Enum):
    # BASIC = "configs/basic.cfg"
    # HEALTH = "configs/health_gathering.cfg"
    # DEATHMATCH = "configs/deathmatch.cfg"
    # DEFEND = "configs/defend_the_center.cfg"
    # WAY_HOME = "configs/my_way_home.cfg"

    BASIC = "DoomBasic-v0"
    HEALTH = "DoomHealthGathering-v0"
    DEATHMATCH = "DoomDeathmatch-v0"
    DEFEND = "DoomDefendCenter-v0"
    WAY_HOME = "DoomMyWayHome-v0"

class Environment(object):
    def __init__(self, level = Level.BASIC, combine_actions = False, visible = True):
        
        # self.game = DoomGame()
        # self.game.load_config(level.value)
        # self.game.set_window_visible(visible)
        # self.game.init()
        # self.actions_num = self.game.get_available_buttons_size()
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
        # self.screen_width = self.game.get_screen_width()
        # self.screen_height = self.game.get_screen_height()

        self.screen_width = self.game.screen_width
        self.screen_height = self.game.screen_height

    def step(self, action):
        # reward = self.game.make_action(action)
        # next_state = self.game.get_state().image_buffer
        # game_over = self.game.is_episode_finished()

        next_state, reward, game_over, _ = self.game.step(action)

        return next_state, reward, game_over

    def get_curr_state(self):
        return self.game.reset()

    def new_episode(self):
        # self.game.new_episode()
        self.game.reset()

    def is_game_over(self):
        return self.game.is_episode_finished()


