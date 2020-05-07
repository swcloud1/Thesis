import paralleldots
import sys

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


def main(args):
    # Setting your API key
    paralleldots.set_api_key("QBDTl86PKSyRGGJDAO9RP3AG2DmhMybT41tUGL0nldE")

    if args:
        print("Input: {}".format(args[0]))
        # print(paralleldots.emotion(args[0]))
        printResults(paralleldots.emotion(args[0])["emotion"])
    else:
        print("no args")



def printResults(results):
    print("Results")
    print("Angry:\t\t{}".format(results["Angry"]))
    print("Bored:\t\t{}".format(results["Bored"]))
    print("Excited:\t{}".format(results["Excited"]))
    print("Fear:\t\t{}".format(results["Fear"]))
    print("Happy:\t\t{}".format(results["Happy"]))
    print("Sad:\t\t{}".format(results["Sad"]))

if __name__ == "__main__":
    main(sys.argv[1:])
