import csv
import sys
import string
import nltk

from itertools import chain
from itertools import product
from difflib import get_close_matches as gcm

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import random
import language_tool_python

class FeatureFlags():
    debug = False
    check_grammar = False
    test_detection_accuracy = False
    progress = True

intensity_lexicon = []
# ekman_emotions = [
# "anger",
# "disgust",
# "fear",
# "joy",
# "sadness",
# "surprise"]
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




class GrammarCheck():

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.tool = language_tool_python.LanguageTool('en-US')

    def check(self, sentence):
        if self.enabled:
            matches = self.tool.check(sentence)
            if matches:
                print("Grammar Check: Has Errors")
            else:
                print("Grammar Check: No Errors")
            for match in matches:
                print(match.ruleId, match.replacements)



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

class ValidationSentence():

    def __init__(self, text, emotion):
        self.text = text
        self.emotion = emotion

def main(args):
    global intensity_lexicon


    intensity_lexicon = loadIntensityLexicon('emotionintensitylexicon.txt')
    input_sentences = load_validation_sentences('rulebasedandmlbasedworking.txt', amount = None)
    progress = Progress(len(input_sentences), enabled=FeatureFlags().progress)
    validator = TestDetection(len(input_sentences), enabled=FeatureFlags().test_detection_accuracy)
    grammar_tool = GrammarCheck(enabled=FeatureFlags().check_grammar)

    if not args:
        for input_sentence in input_sentences:
            progress.print_progress()

            print("\nInput: {}".format(input_sentence.text))
            grammar_tool.check(input_sentence.text)

            preprocessed_input = preprocess(input_sentence.text)
            analysed_sentence = analyse(preprocessed_input)
            print("Analysed Sentence: {}".format(analysed_sentence.emotions))
            validator.validate(input_sentence, analysed_sentence)

            intensified_sentence = intensify(analysed_sentence)
            print("Intensified Sentence: {}".format(intensified_sentence))
            grammar_tool.check(intensified_sentence)

            less_intensified_sentence = lessen(analysed_sentence)
            print("Less-Intensified Sentence: {}".format(less_intensified_sentence))
            grammar_tool.check(less_intensified_sentence)

        validator.print_results()

    else:
        input = args[0]
        print("Input: {}".format(input))
        preprocessed_input = preprocess(input)
        print(preprocessed_input)
        analysed_sentence = analyse(preprocessed_input)
        intensified_sentence = intensify(analysed_sentence)
        print("Output: {}".format(intensified_sentence))


def intensify(input):
    transformer = Transformer()
    return transformer.intensify(input, "more")

def lessen(input):
    transformer = Transformer()
    return transformer.intensify(input, "less")

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

    for word in input:
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

class Transformer:

    global intensity_lexicon

    def intensify(self, sentence, direction):
        replacements = {}
        contributingWords = self.getContributingWords(sentence)
        for word in contributingWords:
            if direction == "more":
                replacements[word.word] = self.getMoreIntenseWord(word)
            if direction == "less":
                replacements[word.word] = self.getLessIntenseWord(word)

        output = self.replace_words(sentence, replacements)
        return "{}".format(' '.join(output))

    def replace_words(self, sentence, replacements):
        output = []
        for word in sentence.words:
            if word.word in replacements:
                output.append(replacements[word.word])
            else:
                output.append(word.word)
        return output

    def getContributingWords(self, sentence):
        contributingWords = []
        for word in sentence.words:
            if word.hasEmotion:
                if word.getMainEmotion()[0] == sentence.getMainEmotion()[0]:
                    contributingWords.append(word)

        return contributingWords

    def getMoreIntenseWord(self, word):
        emotion = word.getMainEmotion()
        print("Replacing word {}({}) with emotion {} with a more intense word".format(word.word, word.pos, emotion))
        matching_words = [x for x in intensity_lexicon if x[1] == emotion[0] and x[2] > emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def getLessIntenseWord(self, word):

        emotion = word.getMainEmotion()
        print("Replacing word {}({}) with emotion {} with a less intense word".format(word.word, word.pos, emotion))
        matching_words = [x for x in intensity_lexicon if x[1] == emotion[0] and x[2] < emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)


    def synonyms(self, word):
        synonyms = []

        for syn in wordnet.synsets(word, pos=wordnet.ADJ):
            for l in syn.lemmas():
                synonyms.append(l.name())

        return synonyms


    def getMostFittingReplacementWord(self,source_word, matching_words):

        best_match = ""
        highest_score = 0.0

        # get meanings of word
        source_wordsets = self.get_wordsets(source_word.word, source_word.pos)
        # cycle through source word meanings
        for source_wordset in source_wordsets:
            # cycle through matching target words
            for matching_word in matching_words:
                # get meanings of matching target word
                matching_wordsets = self.get_wordsets(matching_word[0], source_word.pos)
                # cycle through meanings of matching target word
                synonyms = []
                for matching_wordset in matching_wordsets:
                    # for l in matching_wordset.lemmas():
                    #     synonyms.append(l.name())
                    name = matching_wordset.lemmas()[0].name()
                    # if the synonym is found in the lexicon
                    if name == matching_word[0]:
                        # check the similarty to original word
                        similarity = source_wordset.wup_similarity(matching_wordset)
                        # if it is the current highest similarity store its name and similarity
                        if similarity and similarity > highest_score:
                            highest_score = similarity
                            best_match = name

        if best_match != "":
            dbprint("found best match {} with score {}".format(best_match, highest_score))
            return best_match
        if matching_words:
            replacement_word = random.choice(matching_words)
            dbprint("found replacement word {}".format(replacement_word))
            return replacement_word[0]
        else:
            dbprint("found no replacement word")
            return source_word.word


    def get_wordsets(self, word, pos):
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

        if pos in posdict:
            return wordnet.synsets(word, pos=posdict[pos])
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
            self.cur_percentage = int(self.cur_trial/self.total_trials*100)
            if (self.cur_percentage % self.increment == 0) and (self.cur_percentage != self.prev_percentage):
                print("\nCompletion: {}%".format(self.cur_percentage))
                self.prev_percentage = self.cur_percentage
            self.cur_trial+=1

def dbprint(input):
    if FeatureFlags().debug:
        print(input)

if __name__ == "__main__":
    main(sys.argv[1:])
