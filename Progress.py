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
