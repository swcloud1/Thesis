import paralleldots
import sys
import csv
from time import sleep
# from watson_developer_cloud import ToneAnalyzerV3

# import json
# import os
# from os.path import join
# from ibm_watson import ToneAnalyzerV3
# from ibm_watson.tone_analyzer_v3 import ToneInput
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

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

    # authenticator = IAMAuthenticator('33POoAT_BONb6uokausdJTso7PA9tuLQMg609J7A4P2x')
    # service = ToneAnalyzerV3(
    #     version='2017-09-21',
    #     authenticator=authenticator)
    # service.set_service_url('https://api.eu-de.tone-analyzer.watson.cloud.ibm.com/instances/61edcfea-c654-4825-bfaa-cd6b2dd4ac02')



    # Setting your API key
    paralleldots.set_api_key("QBDTl86PKSyRGGJDAO9RP3AG2DmhMybT41tUGL0nldE")
    # paralleldots.set_api_key("eLnPxnIN91jcilqJl3wkBIYC6nl8DEwIFY5RLdjoCKs")
    # paralleldots.set_api_key("EmbOIWaqS5XqSzbuJ95QRRrS0cL4tQOlredwy8JniDY")
    # paralleldots.set_api_key("jCon8ZO3OYuk3w5FwnzxgRPt3wlIdFgQueDfP24sOzs")
    #
    #
    # #
    # #
    # test_results = {
    #     "anger":{"total":0,"correct":0},
    #     "fear":{"total":0,"correct":0},
    #     "fear":{"total":0,"correct":0},
    #     "sadness":{"total":0,"correct":0}
    # }

    #
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
