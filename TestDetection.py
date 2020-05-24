class TestDetection():

    def __init__(self, test_size, enabled=True):
        self.enabled = enabled
        self.test_size = test_size
        self.matches = 0
        self.test_results = {
            "anger":{"total":0,"correct":0},
            "fear":{"total":0,"correct":0},
            "sadness":{"total":0,"correct":0},
            "joy":{"total":0,"correct":0}
        }

    def validate(self, input_sentence, output_sentence):
        if self.enabled:
            self.test_results[input_sentence.emotion]["total"] += 1
            if output_sentence.getMainEmotion()[0] == input_sentence.emotion:
                print("Match: Emotion={}".format(analysed_sentence.getMainEmotion()[0]))
                self.matches += 1
                self.test_results[input_sentence.emotion]["correct"] += 1
            else:
                print("Not Match: RB-Emotion={}, VAL-Emotion={}".format(analysed_sentence.getMainEmotion()[0], input_sentence.emotion))

    def print_results(self):
        if self.enabled:
            print("\nCorrectness: {}/{} = {}%".format(self.matches, self.test_size, (self.matches/self.test_size*100)))
            print("Test Result: {}".format(self.test_results))
