import csv
import sys
import string
import nltk
import time


from itertools import chain
from itertools import product
from difflib import get_close_matches as gcm

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import random


from ValidationSentence import ValidationSentence
from GrammarCheck import GrammarCheck
from FeatureFlags import FeatureFlags
from AdjustmentType import AdjustmentType
from AdjustmentValidator import AdjustmentValidator
from TestDetection import TestDetection


intensity_lexicon = []
ekman_emotions = [
"anger",
"fear",
"joy",
"sadness"]


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
    return [x for x in sentences if x.emotion in ekman_emotions]







# class ValidationSentence():
#
#     def __init__(self, text, emotion):
#         self.text = text
#         self.emotion = emotion


def main(args):
    global intensity_lexicon


    intensity_lexicon = loadIntensityLexicon('emotionintensitylexicon.txt')
    input_sentences = load_validation_sentences('rulebasedandmlbasedworking.txt', amount = 5)
    progress = Progress(len(input_sentences), enabled=FeatureFlags().progress)
    detection_validator = TestDetection(len(input_sentences), enabled=FeatureFlags().test_detection_accuracy)
    adjustment_validator = AdjustmentValidator()
    transformer = Transformer()
    grammar_tool = GrammarCheck(enabled=FeatureFlags().check_grammar)

    if not args:

        test_results = {
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

        for input_sentence in input_sentences:
            progress.print_progress()

            print("Input Sentence: {}".format(input_sentence.text))
            grammar_tool.check(input_sentence.text)

            input_sentence_a = analyse(input_sentence.text)
            print("\tScore: {}".format(input_sentence_a.getRoundedEmotions()))
            detection_validator.validate(input_sentence, input_sentence_a)


            """
            Adjusting: Replacing Specific Emotion
            """
            replaced_sentence =  transformer.replace(input_sentence_a, AdjustmentType.REPLACE, source_emotion="sadness", target_emotion="fear")
            replaced_sentence_a = analyse(replaced_sentence)
            print("Replaced Sentence: {}".format(replaced_sentence))
            print("\tScore: {}".format(replaced_sentence_a.getRoundedEmotions()))
            # test_result = adjustment_validator.validate(input_sentence_a, more_intense_sentence_a, AdjustmentType.MORE_INTENSE_MAIN)
            # test_results["more_intense"]["total"] += 1
            # test_results["more_intense"]["correct"] += test_result
            # test_results["more_intense"][input_sentence.emotion]["total"] += 1
            # test_results["more_intense"][input_sentence.emotion]["correct"] += test_result
            # grammar_tool.check(more_intense_sentence)

            """
            Adjustment: Altering Specific Emotion Intensity
            """
            # for emotion in ekman_emotions:
            #     more_intense_sentence =  transformer.intensify(input_sentence_a, AdjustmentType.MORE_INTENSE_SPECIFIC, emotion)
            #     more_intense_sentence_a = analyse(more_intense_sentence)
            #     print("More {} Sentence: {}".format(emotion, more_intense_sentence))
            #     print("\tScore: {}".format(more_intense_sentence_a.getRoundedEmotions()))
            #     test_result = adjustment_validator.validate(input_sentence_a, more_intense_sentence_a, AdjustmentType.MORE_INTENSE_SPECIFIC, emotion)
            #     if input_sentence_a.emotions[emotion] > 0:
            #         test_results["more_intense"][emotion]["total"] += 1
            #         test_results["more_intense"][emotion]["correct"] += test_result
            #
            #     less_intense_sentence =  transformer.intensify(input_sentence_a, AdjustmentType.LESS_INTENSE_SPECIFIC, emotion)
            #     less_intense_sentence_a = analyse(less_intense_sentence)
            #     print("Less {} Sentence: {}".format(emotion, less_intense_sentence))
            #     print("\tScore: {}".format(less_intense_sentence_a.getRoundedEmotions()))
            #     test_result = adjustment_validator.validate(input_sentence_a, less_intense_sentence_a, AdjustmentType.LESS_INTENSE_SPECIFIC, emotion)
            #     if input_sentence_a.emotions[emotion] > 0:
            #         test_results["less_intense"][emotion]["total"] += 1
            #         test_results["less_intense"][emotion]["correct"] += test_result

            """
            Adjustment: Altering Main Emotion Intensity
            """
            # more_intense_sentence =  transformer.intensify(input_sentence_a, AdjustmentType.MORE_INTENSE_MAIN)
            # more_intense_sentence_a = analyse(more_intense_sentence)
            # print("Intensified Sentence: {}".format(more_intense_sentence))
            # print("\tScore: {}".format(more_intense_sentence_a.getRoundedEmotions()))
            # test_result = adjustment_validator.validate(input_sentence_a, more_intense_sentence_a, AdjustmentType.MORE_INTENSE_MAIN)
            # test_results["more_intense"]["total"] += 1
            # test_results["more_intense"]["correct"] += test_result
            # test_results["more_intense"][input_sentence.emotion]["total"] += 1
            # test_results["more_intense"][input_sentence.emotion]["correct"] += test_result
            # grammar_tool.check(more_intense_sentence)

            # less_intense_sentence = transformer.intensify(input_sentence_a, AdjustmentType.LESS_INTENSE_MAIN)
            # less_intense_sentence_a = analyse(less_intense_sentence)
            # print("Less-Intensified Sentence: {}".format(less_intense_sentence))
            # print("\tScore: {}".format(less_intense_sentence_a.getRoundedEmotions()))
            # test_result = adjustment_validator.validate(input_sentence_a, less_intense_sentence_a, AdjustmentType.LESS_INTENSE_MAIN)
            # test_results["less_intense"]["total"] += 1
            # test_results["less_intense"]["correct"] += test_result
            # test_results["less_intense"][input_sentence.emotion]["total"] += 1
            # test_results["less_intense"][input_sentence.emotion]["correct"] += test_result
            # grammar_tool.check(less_intense_sentence)

        detection_validator.print_results()
        print("More Intense: {}".format(test_results["more_intense"]))
        print("Less Intense: {}".format(test_results["less_intense"]))

    else:
        input = args[0]
        print("Input: {}".format(input))
        preprocessed_input = preprocess(input)
        print(preprocessed_input)
        analysed_sentence = analyse(preprocessed_input)
        intensified_sentence = intensify(analysed_sentence)
        print("Output: {}".format(intensified_sentence))

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


class AnalysedSentence:

    def __init__(self):
        self.words = []
        self.hasEmotion = False
        self.emotions = {}
        for emotion in ekman_emotions:
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

    def getRoundedEmotions(self):
        rounded_emotions = {}
        for emotion in self.emotions:
            rounded_emotions[emotion] = round(self.emotions[emotion], 3)
        return rounded_emotions

class Transformer:

    global intensity_lexicon

    posdict = {
        "VB":wordnet.VERB,
        "VBD":wordnet.VERB,
        "VBG":wordnet.VERB,
        "VBN":wordnet.VERB,
        "VBP":wordnet.VERB,
        "VBZ":wordnet.VERB,
        "NN":wordnet.NOUN,
        "NNP":wordnet.NOUN,
        "NNS":wordnet.NOUN,
        "JJ":wordnet.ADJ,
        "JJR":wordnet.ADJ,
        "JJS":wordnet.ADJ,
        "RB":wordnet.ADV,
        "RBR":wordnet.ADV,
        "RBS":wordnet.ADV
    }

    def intensify(self, sentence, adjustment_type, emotion=None):
        replacements = {}

        if adjustment_type == AdjustmentType.MORE_INTENSE_MAIN:
            contributingWords = self.getContributingWords(sentence, sentence.getMainEmotion()[0])
            for word in contributingWords:
                replacements[word.word] = self.getMoreIntenseMainEmotionWord(word)

        if adjustment_type == AdjustmentType.LESS_INTENSE_MAIN:
            contributingWords = self.getContributingWords(sentence, sentence.getMainEmotion()[0])
            for word in contributingWords:
                replacements[word.word] = self.getLessIntenseMainEmotionWord(word)

        if adjustment_type == AdjustmentType.MORE_INTENSE_SPECIFIC:
            contributingWords = self.getContributingWords(sentence, emotion)
            for word in contributingWords:
                replacements[word.word] = self.getMoreIntenseSpecificEmotionWord(word, emotion)

        if adjustment_type == AdjustmentType.LESS_INTENSE_SPECIFIC:
            contributingWords = self.getContributingWords(sentence, emotion)
            for word in contributingWords:
                replacements[word.word] = self.getLessIntenseSpecificEmotionWord(word, emotion)

        output = self.replace_words(sentence, replacements)
        return "{}".format(' '.join(output))

    def replace(self, sentence, adjustment_type, source_emotion, target_emotion):
        replacements = {}

        if adjustment_type == AdjustmentType.REPLACE:
            contributingWords = self.getContributingWords(sentence, source_emotion)
            for word in contributingWords:
                replacements[word.word] = self.getEqualIntensWithDifferentEmotion(word, source_emotion, target_emotion)

        output = self.replace_words(sentence, replacements)
        return "{}".format(' '.join(output))

    def getEqualIntensWithDifferentEmotion(self, word, source_emotion, target_emotion):
        words_with_target = [x[0] for x in intensity_lexicon if x[1] == target_emotion]
        words_with_source = [x[0] for x in intensity_lexicon if x[1] == source_emotion]
        matching_words = [x for x in words_with_target if x not in words_with_source]

        return self.getMostFittingReplacementWord(word, matching_words)

    def replace_words(self, sentence, replacements):
        output = []
        for word in sentence.words:
            if word.word in replacements:
                output.append(replacements[word.word])
            else:
                output.append(word.word)
        return output

    def getContributingWords(self, sentence, emotion):
        contributingWords = []
        for word in sentence.words:
            if word.hasEmotion:
                if emotion in word.emotions:
                    contributingWords.append(word)

        return contributingWords

    def getMoreIntenseMainEmotionWord(self, word):
        emotion = word.getMainEmotion()
        dbprint("Replacing word {}({}) with emotion {} with a more intense word".format(word.word, word.pos, emotion))
        matching_words = [x[0] for x in intensity_lexicon if x[1] == emotion[0] and x[2] > emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def getLessIntenseMainEmotionWord(self, word):

        emotion = word.getMainEmotion()
        dbprint("Replacing word {}({}) with emotion {} with a less intense word".format(word.word, word.pos, emotion))
        matching_words = [x[0] for x in intensity_lexicon if x[1] == emotion[0] and x[2] < emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def getMoreIntenseSpecificEmotionWord(self, word, emotion):
        dbprint("Replacing word {}({}) with emotion {} with a more intense word".format(word.word, word.pos, emotion))
        emotion = word.getEmotion(emotion)
        matching_words = [x[0] for x in intensity_lexicon if x[1] == emotion[0] and x[2] > emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def getLessIntenseSpecificEmotionWord(self, word, emotion):
        dbprint("Replacing word {}({}) with emotion {} with a more intense word".format(word.word, word.pos, emotion))
        emotion = word.getEmotion(emotion)
        matching_words = [x[0] for x in intensity_lexicon if x[1] == emotion[0] and x[2] < emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def synonyms(self, word, pos):
        synonyms = [word]

        # if pos in self.posdict:
        #     for syn in wordnet.synsets(word, pos=self.posdict[pos]):
        #         for l in syn.lemmas():
        #             synonyms.append(l.name())
        # else:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())

        return synonyms

    def getLemmas(self, synset):
        vars = []
        for l in synset.lemmas():
            vars.append(l.name())
        return vars

    def getMostFittingReplacementWord(self, source_word, matching_words):
        matching_similar_words = {}
        source_word_synsets = self.get_wordsets(source_word.word, source_word.pos)
        for source_word_synset in source_word_synsets:
            source_word_synset_vars = self.getLemmas(source_word_synset)
            for source_word_synset_var in source_word_synset_vars:
                source_word_synset_var_synsets = self.get_wordsets(source_word_synset_var, source_word.pos)
                for source_word_synset_var_synset in source_word_synset_var_synsets:
                    source_word_synset_var_synset_vars = self.getLemmas(source_word_synset_var_synset)
                    for source_word_synset_var_synset_var in source_word_synset_var_synset_vars:
                        for matching_word in matching_words:
                            if source_word_synset_var_synset_var in self.synonyms(matching_word, source_word.pos):
                                similarity = source_word_synset.wup_similarity(source_word_synset_var_synset)
                                if similarity and source_word_synset_var_synset_var != source_word.word:
                                    matching_similar_words[matching_word] = similarity

        if matching_similar_words:
            # print("found similar words to {}: {}".format(source_word.word, matching_similar_words))
            return self.getHighestSimilarity(matching_similar_words)[0]
        #
        # if matching_words:
        #     return random.choice(matching_words)
        else:
            # print("found no similar words, returning source: {}".format(source_word.word))
            return source_word.word

    def getHighestSimilarity(self, matching_similar_words):
        word = max(matching_similar_words, key=matching_similar_words.get)
        similarity = max(matching_similar_words.values())
        return (word,similarity)

    def get_wordsets(self, word, pos):

        if pos in self.posdict:
            return wordnet.synsets(word, pos=self.posdict[pos])
        else:
            return wordnet.synsets(word)

class Progress():

    def __init__(self, total_trials, enabled=True, increment = 5):
        self.enabled = enabled
        self.cur_trial = 0
        self.total_trials = total_trials
        self.cur_percentage = 0
        self.prev_percentage = 0
        self.increment = increment

    def print_progress(self):
        if self.enabled:
            # print("\n({}/{}){}".format(self.cur_trial, self.total_trials, "-"*50))
            self.cur_percentage = int(self.cur_trial/self.total_trials*100)
            if (self.cur_percentage % self.increment == 0) and (self.cur_percentage != self.prev_percentage):
                print("\nCompletion: {}%".format(self.cur_percentage))
                self.prev_percentage = self.cur_percentage
            self.cur_trial+=1

def dbprint(input):
    if FeatureFlags().debug:
        print(input)

if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    print("--- %s seconds ---" % (time.time() - start_time))
