from FeatureFlags import FeatureFlags

class Tools():
    def dbprint(self, input):
        if FeatureFlags().debug:
            print(input)
