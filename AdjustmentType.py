from enum import Enum

class AdjustmentType(Enum):
    MORE_INTENSE_MAIN = 1
    LESS_INTENSE_MAIN = 2
    MORE_INTENSE_SPECIFIC = 3
    LESS_INTENSE_SPECIFIC = 4
    REPLACE = 5
    REMOVE = 6

    FOCUS_SR = 7
    FOCUS_EM = 8
