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

debug = 0
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



def load_sentences(amount=-1):
    sentences = []
    with open('rulebasedandmlbasedworking.txt') as csv_file:
    # with open('validationsentences.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(sentences) == amount:
                break
            sentences.append((row[0],row[1]))
    return [x for x in sentences if x[1] in ekman_emotions]



def main(args):

    global intensity_lexicon

    intensity_lexicon = loadIntensityLexicon('emotionintensitylexicon.txt')
    sentences = load_sentences(amount = 5)
    length_sentences = len(sentences)

    matches = 0
    trial = 0
    prev_percentage = 0
    cur_percentage = 0

    test_results = {
        "anger":{"total":0,"correct":0},
        "fear":{"total":0,"correct":0},
        "sadness":{"total":0,"correct":0},
        "joy":{"total":0,"correct":0}
    }

    if not args:
        for sentence in sentences:
            print()
            cur_percentage = int(trial/length_sentences*100)
            if (cur_percentage % 5 == 0) and (cur_percentage != prev_percentage):
                print("{}%".format(cur_percentage))
                prev_percentage = cur_percentage
            trial+=1
            input = sentence[0]
            print("Input: {}".format(input))
            preprocessed_input = preprocess(input)
            analysed_sentence = analyse(preprocessed_input)
            # print("RB-Output: {}".format(analysed_sentence.emotions))
            # print("RB-MainEmotion: {}".format(analysed_sentence.getMainEmotion()[0]))
            # print("VAL-Output: {}".format(sentence[1]))
            test_results[sentence[1]]["total"] += 1
            if analysed_sentence.getMainEmotion()[0] == sentence[1]:
                matches += 1
                test_results[sentence[1]]["correct"] += 1
            intensified_sentence = intensify(analysed_sentence)
            print("Intensified Sentence: {}".format(intensified_sentence))
            less_intensified_sentence = lessen(analysed_sentence)
            print("Less-Intensified Sentence: {}".format(less_intensified_sentence))
        print("\nCorrectness: {}/{} = {}%".format(matches, len(sentences), (matches/len(sentences)*100)))
        print("Test Result: {}".format(test_results))

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
        # print(word)
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

    # def printEmotions(self):
    #     print(self.emotions)

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
        # mainEmotion = sentence.getMainEmotion()
        output = []
        replacements = {}
        contributingWords = self.getContributingWords(sentence)
        for word in contributingWords:
            if direction == "more":
                replacements[word.word] = self.getMoreIntenseWord(word)
            if direction == "less":
                replacements[word.word] = self.getLessIntenseWord(word)

        for word in sentence.words:
            if word.word in replacements:
                output.append(replacements[word.word])
            else:
                output.append(word.word)

        return "{}".format(' '.join(output))



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

        print(set(synonyms))
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



def dbprint(input):
    if debug:
        print(input)

if __name__ == "__main__":
    main(sys.argv[1:])
