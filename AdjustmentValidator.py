from AdjustmentType import AdjustmentType
from Emotions import Emotions

class AdjustmentValidator():

    def __init__(self):
        self.intensity_test_results = {
            "more_intense":{
                "total":0,
                "correct":0,
                "anger":{"total":0, "correct":0},
                "fear":{"total":0, "correct":0},
                "joy":{"total":0, "correct":0},
                "sadness":{"total":0, "correct":0}},
            "less_intense":{
                "total":0,
                "correct":0,
                "anger":{"total":0, "correct":0},
                "fear":{"total":0, "correct":0},
                "joy":{"total":0, "correct":0},
                "sadness":{"total":0, "correct":0}}
        }

        self.replace_test_results = {
            "total" : 0,
            "correct" : 0,
            "anger": {"total":0, "anger":0, "fear":0, "joy":0, "sadness":0},
            "fear": {"total":0, "anger":0, "fear":0, "joy":0, "sadness":0},
            "joy": {"total":0, "anger":0, "fear":0, "joy":0, "sadness":0},
            "sadness": {"total":0, "anger":0, "fear":0, "joy":0, "sadness":0}
        }

    def print_validate_replace_results(self):
        print("Total Trials: {}".format(self.replace_test_results["total"]))
        print("Correct Trials: {} = {}%".format(self.replace_test_results["correct"], self.replace_test_results["correct"] / self.replace_test_results["total"] * 100))
        for emotion in Emotions.values():
            print("Total {}: {}".format(emotion, self.replace_test_results[emotion]))

    def validate_replace(self, input_sentence, output_sentence, source_emotion, target_emotion):
        self.replace_test_results["total"] += 1
        self.replace_test_results[source_emotion]["total"] += 1
        output_source = output_sentence.emotions[source_emotion]
        output_target = output_sentence.emotions[target_emotion]
        input_target = input_sentence.emotions[target_emotion]
        if output_source == 0 and output_target > input_target:
            print("\tSucces")
            self.replace_test_results["correct"] += 1
            self.replace_test_results[source_emotion][target_emotion] += 1
        else:
            print("\tFailed")


    def validate(self, input_sentence, output_sentence, adjustment_type, emotion = None):
        if adjustment_type == AdjustmentType.MORE_INTENSE_MAIN:
            input_main_emotion = input_sentence.getMainEmotion()
            output_inputmain_emotion = output_sentence.getEmotion(input_main_emotion[0])

            self.intensity_test_results["more_intense"]["total"] += 1
            self.intensity_test_results["more_intense"][input_main_emotion[0]]["total"] += 1

            if input_main_emotion[0] == output_inputmain_emotion[0] and output_inputmain_emotion[1] > input_main_emotion[1]:
                print("\tSuccessfully made input main emotion {} more intense".format(input_main_emotion[0]))
                self.intensity_test_results["more_intense"]["correct"] += 1
                self.intensity_test_results["more_intense"][input_main_emotion[0]]["correct"] += 1
            else:
                print("\tDid not make input main emotion {} less intense".format(input_main_emotion[0]))
                self.intensity_test_results["more_intense"]["correct"] += 0
                self.intensity_test_results["more_intense"][input_main_emotion[0]]["correct"] += 0

        if adjustment_type == AdjustmentType.LESS_INTENSE_MAIN:
            input_main_emotion = input_sentence.getMainEmotion()
            output_inputmain_emotion = output_sentence.getEmotion(input_main_emotion[0])

            self.intensity_test_results["less_intense"]["total"] += 1
            self.intensity_test_results["less_intense"][input_main_emotion[0]]["total"] += 1

            if input_main_emotion[0] == output_inputmain_emotion[0] and output_inputmain_emotion[1] < input_main_emotion[1]:
                print("\tSuccessfully made input main emotion {} less intense".format(input_main_emotion[0]))
                self.intensity_test_results["less_intense"]["correct"] += 1
                self.intensity_test_results["less_intense"][input_main_emotion[0]]["correct"] += 1
            else:
                print("\tDid not make input main emotion {} less intense".format(input_main_emotion[0]))
                self.intensity_test_results["less_intense"]["correct"] += 0
                self.intensity_test_results["less_intense"][input_main_emotion[0]]["correct"] += 0


        if adjustment_type == AdjustmentType.MORE_INTENSE_SPECIFIC:
            input_emotion = input_sentence.getEmotion(emotion)
            output_input_emotion = output_sentence.getEmotion(input_emotion[0])

            self.intensity_test_results["more_intense"]["total"] += 1
            self.intensity_test_results["more_intense"][emotion]["total"] += 1

            if input_sentence.emotions[emotion] > 0:
                if input_emotion[0] == output_input_emotion[0] and output_input_emotion[1] > input_emotion[1]:
                    print("\tSuccessfully made input emotion {} more intense".format(emotion))
                    self.intensity_test_results["more_intense"]["correct"] += 1
                    self.intensity_test_results["more_intense"][emotion]["correct"] += 1
                else:
                    print("\tDid not make input emotion {} more intense".format(emotion))
                    self.intensity_test_results["more_intense"]["correct"] += 0
                    self.intensity_test_results["more_intense"][emotion]["correct"] += 0

        if adjustment_type == AdjustmentType.LESS_INTENSE_SPECIFIC:
            input_emotion = input_sentence.getEmotion(emotion)
            output_input_emotion = output_sentence.getEmotion(input_emotion[0])

            self.intensity_test_results["less_intense"]["total"] += 1
            self.intensity_test_results["less_intense"][emotion]["total"] += 1

            if input_sentence.emotions[emotion] > 0:
                if input_emotion[0] == output_input_emotion[0] and output_input_emotion[1] < input_emotion[1]:
                    print("\tSuccessfully made input emotion {} less intense".format(emotion))
                    self.intensity_test_results["less_intense"]["correct"] += 1
                    self.intensity_test_results["less_intense"][emotion]["correct"] += 1
                else:
                    print("\tDid not make input emotion {} less intense".format(emotion))
                    self.intensity_test_results["less_intense"]["correct"] += 0
                    self.intensity_test_results["less_intense"][emotion]["correct"] += 0
