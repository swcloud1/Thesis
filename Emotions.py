from enum import Enum

class Emotions(Enum):
    ANGER = "anger"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"

    def values():
        return [item.value for item in Emotions]
