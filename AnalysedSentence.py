from Emotions import Emotions

class AnalysedSentence:

    def __init__(self):
        self.words = []
        self.hasEmotion = False
        self.emotions = {}
        for emotion in Emotions.values():
            self.emotions[emotion] = 0.0

    def feed(self, word):
        self.words.append(word)
        if word.hasEmotion:
            self.hasEmotion = True

    def analyseEmotions(self):
        for word in self.words:
            for emotion in word.emotions:
                if emotion in self.emotions:
                    self.emotions[emotion] += word.emotions[emotion]
        return self.emotions

    def getMainEmotion(self):
        emotion = max(self.emotions, key=self.emotions.get)
        value = max(self.emotions.values())
        if value > 0:
            return (emotion,value)
        else:
            return (None, value)

    def getEmotion(self, emotion):
        return (emotion,self.emotions[emotion])

    def hasEmotion(self, emotion):
        print(self.emotions)
        return self.emotions[emotion] > 0.0

    def getRoundedEmotions(self):
        rounded_emotions = {}
        for emotion in self.emotions:
            rounded_emotions[emotion] = round(self.emotions[emotion], 3)
        return rounded_emotions
