import paralleldots
import sys
import csv
from time import sleep

# data = [ "I like walking in the park", "Don't repeat the same thing over and over!", "This new Liverpool team is not bad", "I have a throat infection" ]



# Viewing your API key
# paralleldots.get_api_key()

# Angry
# Excited
# Sad
# Fear
# Bored
# Happy
#
# print("Emotion\nText: has anyone really been far even as decided to use even go want to do look more like")
# print(paralleldots.emotion("has anyone really been far even as decided to use even go want to do look more like"))

# print( "\nBatch Emotion" )
# results = paralleldots.batch_emotion(data)
# print(results)

ekman_emotions = [
"anger",
"fear",
"sadness"]

emodict = {"Sad": "sadness", "Fear":"fear","Angry":"anger"}

def load_sentences(amount=-1):
    sentences = []
    with open('emo-dataset/val.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            # if len(sentences) == amount:
            #     break
            sentences.append((row[0],row[1]))
    return [x for x in sentences if x[1] in ekman_emotions]


def main(args):
    # Setting your API key
    paralleldots.set_api_key("QBDTl86PKSyRGGJDAO9RP3AG2DmhMybT41tUGL0nldE")


    test_results = {
        "anger":{"total":0,"correct":0},
        "fear":{"total":0,"correct":0},
        "sadness":{"total":0,"correct":0}
    }


    if args:
        print("Input: {}".format(args[0]))
        # print(paralleldots.emotion(args[0]))
        # printResults(paralleldots.emotion(args[0])["emotion"])
    else:
        sentences = load_sentences(amount=-2)

        start = 650
        limit = 50

        while start < len(sentences):


        # sentences = sentences[100:150]
        # print(len(sentences))
            print("\nSentences [{}:{}]".format(start, start+limit))
            results = paralleldots.batch_emotion([x[0] for x in sentences[start:start+limit]])
            # print(results)

            matches = 0

            for i in range(len(sentences[start:start+limit])):
                # print()
                # print(sentences[start:start+limit][i][0])
                # print(results['emotion'])
                ml_results = filterResult(results['emotion'][i])
                # print(ml_results)
                max_ml_emo = max(ml_results, key=ml_results.get)
                # print("ML: {}".format(max_ml_emo))
                # print("VAL: {}".format(sentences[start:start+limit][i][1]))
                test_results[sentences[start:start+limit][i][1]]["total"] += 1
                if max_ml_emo == sentences[start:start+limit][i][1]:
                    matches += 1
                    test_results[sentences[start:start+limit][i][1]]["correct"] += 1

            print("\nCorrectness: {}/{} = {}%".format(matches, len(sentences[start:start+limit]), (matches/len(sentences[start:start+limit])*100)))
            print("Test Result: {}".format(test_results))

            start += limit
            for i in range(12):
                print("Sleeping: {}".format(i * 5))
                sleep(5)


def filterResult(result):
    emotions = {}
    # for emotion in ekman_emotions:
    #     emotions[emotion] = 0.0

    for emotion in result:
        if emotion in emodict:
            emotions[emodict[emotion]] = result[emotion]

    return emotions



def printResults(results):
    print("Results")
    print("Angry:\t\t{}".format(results["Angry"]))
    # print("Bored:\t\t{}".format(results["Bored"]))
    # print("Excited:\t{}".format(results["Excited"]))
    print("Fear:\t\t{}".format(results["Fear"]))
    # print("Happy:\t\t{}".format(results["Happy"]))
    print("Sad:\t\t{}".format(results["Sad"]))

if __name__ == "__main__":
    main(sys.argv[1:])
