class Hypothesis(object):
    def __init__(self, tokens, state, log_probs):
        self.tokens =  tokens
        self.state = state
        self.log_probs = log_probs

    def extend(self, token, state, log_prob):
        return Hypothesis(tokens = self.tokens + [token],
                          log_probs = self.log_probs + [log_prob],
                          state = state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.log_prob/len(self.tokens)
