import paralleldots
import sys
import csv
from time import sleep

ekman_emotions = [
"anger",
"fear",
"joy",
"sadness"]

emodict = {"Sad": "sadness", "Fear":"fear","Angry":"anger", "Happy":"joy"}

def load_sentences(amount=-1):
    sentences = []
    # with open('emo-dataset/val.txt') as csv_file:
    with open('rulebasedworking.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if len(sentences) == amount:
                break
            sentences.append((row[0],row[1]))
    return [x for x in sentences if x[1] in ekman_emotions]


def main(args):
    # Setting your API key
    paralleldots.set_api_key("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    if args:
        print("Input: {}".format(args[0]))
        print(paralleldots.emotion(args[0]))
        printResults(paralleldots.emotion(args[0])["emotion"])
    else:
        sentences = load_sentences(amount=-2)

        start = 50
        limit = 50

        while start < len(sentences):

            test_results = {
                "anger":{"total":0,"correct":0},
                "fear":{"total":0,"correct":0},
                "sadness":{"total":0,"correct":0},
                "joy":{"total":0,"correct":0}
            }

            print("\nSentences [{}:{}]".format(start, start+limit))
            results = paralleldots.batch_emotion([x[0] for x in sentences[start:start+limit]])
            matches = 0

            for i in range(len(sentences[start:start+limit])):
                ml_results = filterResult(results['emotion'][i])
                max_ml_emo = max(ml_results, key=ml_results.get)
                test_results[sentences[start:start+limit][i][1]]["total"] += 1
                if max_ml_emo == sentences[start:start+limit][i][1]:
                    matches += 1
                    test_results[sentences[start:start+limit][i][1]]["correct"] += 1
                    write_sentence_to_file(sentences[start:start+limit][i])


            print("Correctness: {}/{} = {}%".format(matches, len(sentences[start:start+limit]), (matches/len(sentences[start:start+limit])*100)))
            print("Test Result: {}".format(test_results))

            start += limit
            for i in range(12):
                print("Sleeping: {}".format(i * 5))
                sleep(5)

def write_sentence_to_file(sentence):
    with open('rulebasedandmlbasedworking.txt','a') as fd:
            fd.write("{};{}\n".format(sentence[0], sentence[1]))

def filterResult(result):
    emotions = {}
    for emotion in result:
        if emotion in emodict:
            emotions[emodict[emotion]] = result[emotion]

    return emotions



def printResults(results):
    print("Results")
    print("Angry:\t\t{}".format(results["Angry"]))
    print("Fear:\t\t{}".format(results["Fear"]))
    print("Sad:\t\t{}".format(results["Sad"]))

if __name__ == "__main__":
    main(sys.argv[1:])
