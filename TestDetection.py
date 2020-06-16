"""
/*
 * Copyright 2020 Bloomreach B.V. (http://www.bloomreach.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 """
 
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
