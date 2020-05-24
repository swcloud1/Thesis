import random

from nltk.corpus import wordnet

from Tools import Tools
from AdjustmentType import AdjustmentType
from FeatureFlags import FeatureFlags


class Transformer:

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

    def __init__(self, intensity_lexicon):
        self.intensity_lexicon = intensity_lexicon

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
        words_with_target = [x[0] for x in self.intensity_lexicon if x[1] == target_emotion]
        words_with_source = [x[0] for x in self.intensity_lexicon if x[1] == source_emotion]
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
        Tools().dbprint("Replacing word {}({}) with emotion {} with a more intense word".format(word.word, word.pos, emotion))
        matching_words = [x[0] for x in self.intensity_lexicon if x[1] == emotion[0] and x[2] > emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def getLessIntenseMainEmotionWord(self, word):

        emotion = word.getMainEmotion()
        Tools().dbprint("Replacing word {}({}) with emotion {} with a less intense word".format(word.word, word.pos, emotion))
        matching_words = [x[0] for x in self.intensity_lexicon if x[1] == emotion[0] and x[2] < emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def getMoreIntenseSpecificEmotionWord(self, word, emotion):
        Tools().dbprint("Replacing word {}({}) with emotion {} with a more intense word".format(word.word, word.pos, emotion))
        emotion = word.getEmotion(emotion)
        matching_words = [x[0] for x in self.intensity_lexicon if x[1] == emotion[0] and x[2] > emotion[1] and x[3] == word.pos]

        return self.getMostFittingReplacementWord(word, matching_words)

    def getLessIntenseSpecificEmotionWord(self, word, emotion):
        Tools().dbprint("Replacing word {}({}) with emotion {} with a more intense word".format(word.word, word.pos, emotion))
        emotion = word.getEmotion(emotion)
        matching_words = [x[0] for x in self.intensity_lexicon if x[1] == emotion[0] and x[2] < emotion[1] and x[3] == word.pos]

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

        if FeatureFlags().adjustment_focus == AdjustmentType.FOCUS_SR:
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
                Tools().dbprint("found similar words to {}: {}".format(source_word.word, matching_similar_words))
                return self.getHighestSimilarity(matching_similar_words)[0]

            else:
                Tools().dbprint("found no similar words, returning source: {}".format(source_word.word))
                return source_word.word

        if FeatureFlags().adjustment_focus == AdjustmentType.FOCUS_EM:
            if matching_words:
                random_matching_word = random.choice(matching_words)
                Tools().dbprint("found matching word {}, returning source: {}".format(random_matching_word, source_word.word))
                return random.choice(random_matching_word)
            else:
                Tools().dbprint("found no similar words, returning source: {}".format(source_word.word))
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
