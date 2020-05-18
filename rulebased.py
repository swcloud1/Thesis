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
"sadness"]


def preprocess(text):
    # input = remove_punctuation(text)
    # input = input.split()
    # if " " not in text:
    #     input = [text]
    #
    #
    # words = []
    #
    # # ps = PorterStemmer()
    # for word in input:
    #     words.append(word)
    tokenized_sentence = nltk.word_tokenize(text)
    pos_tagged_sentence = nltk.pos_tag(tokenized_sentence)
    return pos_tagged_sentence


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

def load_sentences(amount=-1):
    sentences = []
    with open('emo-dataset/val.txt') as csv_file:
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
            print("RB-Output: {}".format(analysed_sentence.emotions))
            print("RB-MainEmotion: {}".format(analysed_sentence.getMainEmotion()[0]))
            # print("VAL-Output: {}".format(sentence[1]))
            if analysed_sentence.getMainEmotion()[0] == sentence[1]:
                matches += 1
            intensified_sentence = intensify(analysed_sentence)
            print("Output: {}".format(intensified_sentence))
        # print("\nCorrectness: {}/{} = {}%".format(matches, len(sentences), (matches/len(sentences)*100)))

    else:
        input = args[0]
        print("Input: {}".format(input))
        preprocessed_input = preprocess(input)
        analysed_sentence = analyse(preprocessed_input)
        intensified_sentence = intensify(analysed_sentence)
        print("Output: {}".format(intensified_sentence))


def intensify(input):
    transformer = Transformer()
    return transformer.intensify(input, "more")

def lessen(input):
    transformer = Transformer()
    return transformer.lessen(input, "less")

def loadIntensityLexicon(source):
    intensity_lexicon = []
    with open(source) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None)
        for row in csv_reader:
            intensity_lexicon.append((row[0], row[1], float(row[2])))

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

    def getLessIntenseWord(self, word):
        intenserWords = []
        emotion = word.getMainEmotion()
        with open('emotionintensitylexicon.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if row[1] == emotion[0]:
                    if float(row[2]) < emotion[1]:
                        intenserWords.append(row[0])
        word_type = nltk.pos_tag([word.word])
        tagged_intenser_words = nltk.pos_tag(intenserWords)
        interserMatchingWords = [word for word in tagged_intenser_words if word[1] == word_type[0][1]]
        return random.choice(interserMatchingWords)[0]


    def getMoreIntenseWord(self, word):
        intenser_words = []
        emotion = word.getMainEmotion()
        with open('emotionintensitylexicon.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if row[1] == emotion[0]:
                    if float(row[2]) > emotion[1]:
                        intenser_words.append(row[0])
        tagged_intenser_words = [nltk.pos_tag([x])[0] for x in intenser_words]
        interserMatchingWords = [intenser_word for intenser_word in tagged_intenser_words if intenser_word[1] == word.pos]
        return random.choice(interserMatchingWords)[0]

def dbprint(input):
    if debug:
        print(input)

if __name__ == "__main__":
    main(sys.argv[1:])
