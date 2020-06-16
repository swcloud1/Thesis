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
 
class Progress():

    def __init__(self, total_trials, enabled=True, increment = 5):
        self.enabled = enabled
        self.cur_trial = 0
        self.total_trials = total_trials
        self.cur_percentage = 0
        self.prev_percentage = 0
        self.increment = increment

    def print_progress(self):
        if self.enabled:
            print("\n({}/{}){}".format(self.cur_trial, self.total_trials, "-"*50))
            self.cur_percentage = int(self.cur_trial/self.total_trials*100)
            if (self.cur_percentage % self.increment == 0) and (self.cur_percentage != self.prev_percentage):
                print("\nCompletion: {}%".format(self.cur_percentage))
                self.prev_percentage = self.cur_percentage
            self.cur_trial+=1
