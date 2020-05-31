import csv
import sys
import string
import nltk
import time
import random

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from ValidationSentence import ValidationSentence
from GrammarCheck import GrammarCheck
from FeatureFlags import FeatureFlags
from AdjustmentType import AdjustmentType
from AdjustmentValidator import AdjustmentValidator
from TestDetection import TestDetection
from Progress import Progress
from AnalysedWord import AnalysedWord
from AnalysedSentence import AnalysedSentence
from Emotions import Emotions
from Transformer import Transformer
from Tools import Tools

intensity_lexicon = []

def main(args):
    global intensity_lexicon

    intensity_lexicon = loadIntensityLexicon('emotionintensitylexicon.txt')
    input_sentences = load_validation_sentences('rulebasedandmlbasedworking.txt', amount = None)
    progress = Progress(len(input_sentences), enabled=FeatureFlags().progress)
    detection_validator = TestDetection(len(input_sentences), enabled=FeatureFlags().test_detection_accuracy)
    adjustment_validator = AdjustmentValidator()
    transformer = Transformer(intensity_lexicon)
    grammar_tool = GrammarCheck(enabled=FeatureFlags().check_grammar)

    if not args:
        for input_sentence in input_sentences:
            progress.print_progress()

            print("Input Sentence: {}".format(input_sentence.text))
            grammar_tool.check(input_sentence.text)

            input_sentence_a = analyse(input_sentence.text)
            print("\tScore: {}".format(input_sentence_a.getRoundedEmotions()))
            detection_validator.validate(input_sentence, input_sentence_a)

            """
            Adjustment: Removing Emotion
            """
            # for source_emotion in Emotions.values():
            #     emotion_lower_bound = 0.3
            #     if input_sentence_a.emotions[source_emotion] > emotion_lower_bound:
            #         emotion_removed_sentence =  transformer.remove(input_sentence_a, source_emotion, emotion_lower_bound,  AdjustmentType.REMOVE)
            #         emotion_removed_sentence_a = analyse(emotion_removed_sentence)
            #         print("Removed {}: {}".format(source_emotion, emotion_removed_sentence))
            #         print("\tScore: {}".format(emotion_removed_sentence_a.getRoundedEmotions()))
            #         adjustment_validator.validate_remove(input_sentence_a, emotion_removed_sentence_a, source_emotion, emotion_lower_bound, AdjustmentType.REMOVE)
            # grammar_tool.check(more_intense_sentence)

            """
            Adjusting: Replacing Specific Emotion
            """
            # for source_emotion in Emotions.values():
            #     for target_emotion in Emotions.values():
            #         if source_emotion != target_emotion and input_sentence_a.emotions[source_emotion] > 0:
            #             replaced_sentence =  transformer.replace(input_sentence_a, AdjustmentType.REPLACE, source_emotion=source_emotion, target_emotion=target_emotion)
            #             replaced_sentence_a = analyse(replaced_sentence)
            #             print("Replacing {} with {}".format(source_emotion, target_emotion))
            #             print("Replaced Sentence: {}".format(replaced_sentence))
            #             print("\tScore: {}".format(replaced_sentence_a.getRoundedEmotions()))
            #             adjustment_validator.validate_replace(input_sentence_a, replaced_sentence_a, source_emotion, target_emotion)
            #             grammar_tool.check(replaced_sentence)

            """
            Adjustment: Altering Specific Emotion Intensity
            """
            # for emotion in Emotions.values():
            #     more_intense_sentence =  transformer.intensify(input_sentence_a, AdjustmentType.MORE_INTENSE_SPECIFIC, emotion)
            #     more_intense_sentence_a = analyse(more_intense_sentence)
            #     print("More {} Sentence: {}".format(emotion, more_intense_sentence))
            #     print("\tScore: {}".format(more_intense_sentence_a.getRoundedEmotions()))
            #     adjustment_validator.validate(input_sentence_a, more_intense_sentence_a, AdjustmentType.MORE_INTENSE_SPECIFIC, emotion)
            #
            #     less_intense_sentence =  transformer.intensify(input_sentence_a, AdjustmentType.LESS_INTENSE_SPECIFIC, emotion)
            #     less_intense_sentence_a = analyse(less_intense_sentence)
            #     print("Less {} Sentence: {}".format(emotion, less_intense_sentence))
            #     print("\tScore: {}".format(less_intense_sentence_a.getRoundedEmotions()))
            #     adjustment_validator.validate(input_sentence_a, less_intense_sentence_a, AdjustmentType.LESS_INTENSE_SPECIFIC, emotion)

            """
            Adjustment: Altering Main Emotion Intensity
            """
            # more_intense_sentence =  transformer.intensify(input_sentence_a, AdjustmentType.MORE_INTENSE_MAIN)
            # more_intense_sentence_a = analyse(more_intense_sentence)
            # print("Intensified Sentence: {}".format(more_intense_sentence))
            # print("\tScore: {}".format(more_intense_sentence_a.getRoundedEmotions()))
            # adjustment_validator.validate(input_sentence_a, more_intense_sentence_a, AdjustmentType.MORE_INTENSE_MAIN)
            # grammar_tool.check(more_intense_sentence)

            # less_intense_sentence = transformer.intensify(input_sentence_a, AdjustmentType.LESS_INTENSE_MAIN)
            # less_intense_sentence_a = analyse(less_intense_sentence)
            # print("Less-Intensified Sentence: {}".format(less_intense_sentence))
            # print("\tScore: {}".format(less_intense_sentence_a.getRoundedEmotions()))
            # adjustment_validator.validate(input_sentence_a, less_intense_sentence_a, AdjustmentType.LESS_INTENSE_MAIN)
            # grammar_tool.check(less_intense_sentence)

        # detection_validator.print_results()
        adjustment_validator.print_validate_replace_results()
        # adjustment_validator.print_validate_remove_results()
        # print("More Intense: {}".format(adjustment_validator.intensity_test_results["more_intense"]))
        # print("Less Intense: {}".format(adjustment_validator.intensity_test_results["less_intense"]))

    else:
        emotion = ""
        type = "analyse"
        nonparamtypes = ["analyse", "moreintensemain", "lessintensemain", "moreintense", "lessintense"]
        if args[0] in types:
            type = args[0]
            input_sentence = args[1:]
        else:
            input_sentence = args


        print("Input: {}".format(input_sentence))
        input_sentence_a = analyse(input_sentence)
        more_intense_sentence = transformer.intensify(input_sentence_a, AdjustmentType.MORE_INTENSE_MAIN)
        print("Output: {}".format(more_intense_sentence))
        return {"input": input_sentence, "output":more_intense_sentence}

def loadIntensityLexicon(source):
    intensity_lexicon = []
    with open(source) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None)
        for row in csv_reader:
            pos = nltk.pos_tag([row[0]])[0][1]
            intensity_lexicon.append((row[0], row[1], float(row[2]), pos))

    return intensity_lexicon

def getemotions(word):

    global intensity_lexicon

    emotions = {}
    for row in intensity_lexicon:
        if row[0] == word.lower():
            emotions[row[1]] = row[2]

    return emotions

def analyse(input):
    analysed_sentence = AnalysedSentence()
    preprocessed_input = preprocess(input)

    for word in preprocessed_input:
        analysedWord = AnalysedWord(word[0], word[1], getemotions(word[0]))
        analysed_sentence.feed(analysedWord)

    analysed_sentence.analyseEmotions()

    return analysed_sentence

def preprocess(text):
    tokenized_sentence = nltk.word_tokenize(text)
    pos_tagged_sentence = nltk.pos_tag(tokenized_sentence)
    return pos_tagged_sentence


def remove_punctuation(text):
    result = text.translate(str.maketrans('','',string.punctuation))
    return result

def load_validation_sentences(source, amount=-1):
    sentences = []
    with open(source) as csv_file:
    # with open('validationsentences.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if amount and len(sentences) == amount:
                break
            sentences.append(ValidationSentence(row[0],row[1]))
    return [x for x in sentences if x.emotion in Emotions.values()]

if __name__ == "__main__":
    # start_time = time.time()
    main(sys.argv[1:])
    # print("--- %s seconds ---" % (time.time() - start_time))
