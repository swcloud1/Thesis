class AdjustmentValidator():

    def validate(self, input_sentence, output_sentence, adjustment_type, emotion = None):
        if adjustment_type == AdjustmentType.MORE_INTENSE_MAIN:
            input_main_emotion = input_sentence.getMainEmotion()
            output_inputmain_emotion = output_sentence.getEmotion(input_main_emotion[0])
            if input_main_emotion[0] == output_inputmain_emotion[0] and output_inputmain_emotion[1] > input_main_emotion[1]:
                print("\tSuccessfully made input main emotion {} more intense".format(input_main_emotion[0]))
                return 1
            else:
                print("\tDid not make input main emotion {} less intense".format(input_main_emotion[0]))
                return 0

        if adjustment_type == AdjustmentType.LESS_INTENSE_MAIN:
            input_main_emotion = input_sentence.getMainEmotion()
            output_inputmain_emotion = output_sentence.getEmotion(input_main_emotion[0])
            if input_main_emotion[0] == output_inputmain_emotion[0] and output_inputmain_emotion[1] < input_main_emotion[1]:
                print("\tSuccessfully made input main emotion {} less intense".format(input_main_emotion[0]))
                return 1
            else:
                print("\tDid not make input main emotion {} less intense".format(input_main_emotion[0]))
                return 0

        if adjustment_type == AdjustmentType.MORE_INTENSE_SPECIFIC:
            input_emotion = input_sentence.getEmotion(emotion)
            output_input_emotion = output_sentence.getEmotion(input_emotion[0])
            if input_emotion[0] == output_input_emotion[0] and output_input_emotion[1] > input_emotion[1]:
                print("\tSuccessfully made input emotion {} more intense".format(emotion))
                return 1
            else:
                print("\tDid not make input emotion {} more intense".format(emotion))
                return 0

        if adjustment_type == AdjustmentType.LESS_INTENSE_SPECIFIC:
            input_emotion = input_sentence.getEmotion(emotion)
            output_input_emotion = output_sentence.getEmotion(input_emotion[0])
            if input_emotion[0] == output_input_emotion[0] and output_input_emotion[1] < input_emotion[1]:
                print("\tSuccessfully made input emotion {} less intense".format(emotion))
                return 1
            else:
                print("\tDid not make input emotion {} less intense".format(emotion))
                return 0
