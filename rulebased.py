import csv
import sys
import string
import nltk

from itertools import chain
from nltk.corpus import wordnet as wn
from difflib import get_close_matches as gcm

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import random

debug = 0

ekman_emotions = [
"anger",
"disgust",
"fear",
"joy",
"sadness",
"surprise"]


def preprocess(text):
    input = remove_punctuation(text)
    input = input.split()
    if " " not in text:
        input = [text]


    words = []

    ps = PorterStemmer()
    for word in input:
        # words.append(ps.stem(word))
        words.append(word)


    # for word in input:
        # processed_word = adv2adj(word)
        #andere dingen
        # words.append(processed_word)

    return words


def remove_punctuation(text):
    result = text.translate(str.maketrans('','',string.punctuation))
    return result

def adv2adj(word):
    possible_adj = []
    for ss in wn.synsets(word):
      for lemmas in ss.lemmas(): # all possible lemmas
          for ps in lemmas.pertainyms(): # all possible pertainyms
              possible_adj.append(ps.name())
    if possible_adj:
        dbprint("Coverted {} to adjective: {}".format(word, possible_adj[0]))
        return possible_adj[0]

    return word



def main(args):

    input = "I am mad"
    if args:
        input = args[0]
    analyse(input)


def getemotions(word):
    emotions = {}
    with open('emotionintensitylexicon.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if row[0] == word.lower():
                # dbprint(row)
                emotions[row[1]] = float(row[2])
    return emotions

def analyse(input):

    print("Input: {}".format(input))
    preprocessed_input = preprocess(input)
    dbprint("Preprocessed input: {}".format(preprocessed_input))

    analysedSentence = AnalysedSentence()

    for word in preprocessed_input:
        analysedWord = AnalysedWord(word, getemotions(word))
        analysedSentence.feed(analysedWord)

    analysedSentence.analyseEmotions()
    analysedSentence.printEmotions()

    transformer = Transformer()
    transformer.intensify(analysedSentence)
    # output = {}
    # for emotion in ekman_emotions:
    #     output[emotion] = 0.0
    #
    #
    #
    # for word in text:
    #     dbprint(word)
    #     emotions = getemotions(word)
    #     # emotions = getemotions(word)
    #     # sentiment = getsentiment(word)
    #     dbprint(emotions)
    #     for emotion in emotions:
    #         if emotion in ekman_emotions:
    #             output[emotion] += emotions[emotion]
    #
    # print("Output: {}".format(output))

def intensify(input):
    return

def lessen(input):
    return

class AnalysedWord:

    hasEmotion = False

    def __init__(self, word, emotions):
        self.word = word
        self.emotions = emotions
        if emotions:
            self.hasEmotion = True

    def getMainEmotion(self):
        emotion = max(self.emotions, key=self.emotions.get)
        value = max(self.emotions.values())
        return (emotion,value)


class AnalysedSentence:

    words = []
    emotions = {
        "anger":0.0,
        "disgust":0.0,
        "fear":0.0,
        "joy":0.0,
        "sadness":0.0,
        "surprise":0.0
    }

    hasEmotion = False

    def feed(self, word):
        self.words.append(word)
        if word.hasEmotion:
            self.hasEmotion = True

    def analyseEmotions(self):
        for word in self.words:
            for emotion in word.emotions:
                self.emotions[emotion] += word.emotions[emotion]
        return self.emotions

    def printEmotions(self):
        print(self.emotions)

    def getMainEmotion(self):
        emotion = max(self.emotions, key=self.emotions.get)
        value = max(self.emotions.values())
        return (emotion,value)


class Transformer:

    # def __init__(self)

    def intensify(self, sentence):
        # mainEmotion = sentence.getMainEmotion()
        output = []
        replacements = {}
        contributingWords = self.getContributingWords(sentence)
        for word in contributingWords:
            replacements[word.word] = self.getIntenserWord(word)

        for word in sentence.words:
            if word.word in replacements:
                output.append(replacements[word.word])
            else:
                output.append(word.word)

        print("output: {}".format(' '.join(output)))


    def getContributingWords(self, sentence):
        contributingWords = []
        for word in sentence.words:
            if word.hasEmotion:
                if word.getMainEmotion()[0] == sentence.getMainEmotion()[0]:
                    contributingWords.append(word)

        return contributingWords

    def getIntenserWord(self, word):
        intenserWords = []
        emotion = word.getMainEmotion()
        with open('emotionintensitylexicon.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if row[1] == emotion[0]:
                    if float(row[2]) > emotion[1]:
                        intenserWords.append(row[0])
        tagged_intenser_words = nltk.pos_tag(intenserWords)
        interserMatchingWords = [word for word in tagged_intenser_words if word[1] == "JJ"]
        return random.choice(interserMatchingWords)[0]




    # def __init__(self, words):
    #     self.words = words
# def getsentiment(word):
#     positive = 0
#     negative = 0
#     hits = 0
#     with open('sentiwordnet.csv') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter='\t')
#         for row in csv_reader:
#             row_word = row[3].split('#')[0]
#             if row_word == word:
#                 dbprint(row)
#                 hits += 1
#                 positive += float(row[1])
#                 negative += float(row[2])
#
#     if positive > negative and positive != 0:
#         return round(positive / hits, 3)
#     if negative > positive and negative != 0:
#         return round(negative / hits, 3)
#
#     return 0.0





# def getemotions(word):
#     emotions = []
#     with open('emotionlexicon.txt') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter='\t')
#         for row in csv_reader:
#             if row[0] == word.lower() and row[2] == '1':
#                 dbprint(row)
#                 emotions.append(row[1])
#     return emotions



def dbprint(input):
    if debug:
        print(input)

if __name__ == "__main__":
    main(sys.argv[1:])
