import language_tool_python
class GrammarCheck():

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.tool = language_tool_python.LanguageTool('en-US')

    def check(self, sentence):
        if self.enabled:
            matches = self.tool.check(sentence)
            if matches:
                print("Grammar Check: Has Errors")
            else:
                print("Grammar Check: No Errors")
            for match in matches:
                print(match.ruleId, match.replacements)
