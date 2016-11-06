from enum import Enum

class Mode(Enum):
    TRAIN = 1
    TEST = 2
    DISPLAY = 3

class Algorithm(Enum):
    DQN = 1
    DDQN = 2
    DRQN = 3

class Architecture(Enum):
    DIRECT = 1
    DUELING = 2
    SEQUENCE = 3

class ExplorationPolicy(Enum):
    E_GREEDY = 1
    SOFTMAX = 2
    SHIFTED_MULTINOMIAL = 3
