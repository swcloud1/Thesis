import csv
import sys
import string

from itertools import chain
from nltk.corpus import wordnet as wn
from difflib import get_close_matches as gcm

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

debug = 1

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
        print(word, " : ", ps.stem(word))
        words.append(ps.stem(word))


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



def analyse(input):

    print("Input: {}".format(input))
    text = preprocess(input)
    print("Preprocessed input: {}".format(text))

    output = {}
    for emotion in ekman_emotions:
        output[emotion] = 0.0

    for word in text:
        dbprint(word)
        emotions = getemotions(word)
        sentiment = getsentiment(word)
        dbprint(emotions)
        for emotion in emotions:
            if emotion in ekman_emotions:
                output[emotion] += sentiment

    print("Output: {}".format(output))

def getsentiment(word):
    positive = 0
    negative = 0
    hits = 0
    with open('sentiwordnet.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            row_word = row[3].split('#')[0]
            if row_word == word:
                dbprint(row)
                hits += 1
                positive += float(row[1])
                negative += float(row[2])

    if positive > negative and positive != 0:
        return round(positive / hits, 3)
    if negative > positive and negative != 0:
        return round(negative / hits, 3)

    return 0.0

def getemotions(word):
    emotions = []
    with open('emotionlexicon.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if row[0] == word.lower() and row[2] == '1':
                dbprint(row)
                emotions.append(row[1])
    return emotions

def dbprint(input):
    if debug:
        print(input)

if __name__ == "__main__":
    main(sys.argv[1:])
