class AnalysedWord:

    hasEmotion = False

    def __init__(self, word, pos, emotions):
        self.word = word
        self.pos = pos
        self.emotions = emotions
        if emotions:
            self.hasEmotion = True

    def getMainEmotion(self):
        emotion = max(self.emotions, key=self.emotions.get)
        value = max(self.emotions.values())
        return (emotion,value)

    def getEmotion(self, emotion):
        return (emotion,self.emotions[emotion])
